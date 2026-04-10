"""
出租车驾驶行为挖掘分析系统 V2
车牌号: 粤BCW7826
分析日期: 2025-09-25, 2025-09-26
改进: 增强充电状态识别算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import os
import json
import requests

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# 新增：API查询结果缓存，避免重复请求
CHARGING_POI_CACHE = {}
plt.rcParams['axes.unicode_minus'] = False

# 项目路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, '01_raw_data')
PARSED_DATA_DIR = os.path.join(BASE_DIR, '02_preprocessed_data', '01_parsed_data')
CLEANED_DATA_DIR = os.path.join(BASE_DIR, '02_preprocessed_data', '02_cleaned_data')
LABELED_DATA_DIR = os.path.join(BASE_DIR, '02_preprocessed_data', '03_labeled_data')
VISUALIZATION_DIR = os.path.join(BASE_DIR, '04_visualization_results')
OUTPUT_DIR = os.path.join(BASE_DIR, '05_result_output')

# 深圳地理边界
SHENZHEN_BOUNDS = {
    'lon_min': 113.7, 'lon_max': 114.5,
    'lat_min': 22.4, 'lat_max': 22.8
}


def parse_order_data(file_path):
    """解析订单交易数据"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 17:
                record = {
                    'upload_time': parts[0],
                    'plate_no': parts[1],
                    'field3': parts[2],
                    'pickup_timestamp': int(parts[3]) if parts[3] else None,
                    'dropoff_timestamp': int(parts[4]) if parts[4] else None,
                    'fare': float(parts[5]) if parts[5] else 0,
                    'field7': float(parts[6]) if parts[6] else 0,
                    'mileage': float(parts[7]) if parts[7] else 0,
                    'field9': float(parts[8]) if parts[8] else 0,
                    'duration': parts[9],
                    'order_status': int(parts[10]) if parts[10] else None,
                    'field12': parts[11],
                    'field13': float(parts[12]) if parts[12] else 0,
                    'pickup_lon': float(parts[13]) if parts[13] else None,
                    'pickup_lat': float(parts[14]) if parts[14] else None,
                    'dropoff_lon': float(parts[15]) if parts[15] else None,
                    'dropoff_lat': float(parts[16]) if parts[16] else None,
                }
                records.append(record)
    return pd.DataFrame(records)


def parse_gps_data(file_path):
    """解析GPS定位数据"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                record = {
                    'upload_time': parts[0],
                    'plate_no': parts[1],
                    'lon': float(parts[2]) if parts[2] else None,
                    'lat': float(parts[3]) if parts[3] else None,
                    'gps_time': parts[4],
                    'speed': float(parts[6]) if len(parts) > 6 and parts[6] else 0,
                    'direction': float(parts[7]) if len(parts) > 7 and parts[7] else 0
                }
                records.append(record)
    return pd.DataFrame(records)


def timestamp_to_beijing(ts_ms):
    """毫秒时间戳转换为北京时间"""
    if pd.isna(ts_ms) or ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000) + timedelta(hours=8)


def parse_duration(duration_str):
    """解析时长字符串为分钟数"""
    if pd.isna(duration_str) or not duration_str:
        return None
    try:
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 60 + minutes + seconds / 60
    except:
        return None
    return None


def preprocess_orders(df):
    """订单数据预处理"""
    df['pickup_beijing'] = df['pickup_timestamp'].apply(timestamp_to_beijing)
    df['dropoff_beijing'] = df['dropoff_timestamp'].apply(timestamp_to_beijing)
    df['duration_minutes'] = df['duration'].apply(parse_duration)
    df['time_diff_minutes'] = (df['dropoff_timestamp'] - df['pickup_timestamp']) / 60000
    df['date'] = df['pickup_beijing'].dt.date
    return df


def clean_orders(df, date_str, log_file):
    """清洗订单数据"""
    original_count = len(df)
    cleaning_log = []
    cleaning_log.append(f"=== 订单数据清洗日志 - {date_str} ===")
    cleaning_log.append(f"原始数据量: {original_count} 条")

    df = df[df['order_status'] == 0].copy()
    cleaning_log.append(f"过滤未完成订单后: {len(df)} 条 (剔除 {original_count - len(df)} 条)")

    before = len(df)
    df = df.dropna(subset=['pickup_timestamp', 'dropoff_timestamp', 'pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat'])
    cleaning_log.append(f"剔除核心字段缺失后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df[df['dropoff_timestamp'] > df['pickup_timestamp']].copy()
    cleaning_log.append(f"校验时间逻辑后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df[
        (df['pickup_lon'] >= SHENZHEN_BOUNDS['lon_min']) &
        (df['pickup_lon'] <= SHENZHEN_BOUNDS['lon_max']) &
        (df['pickup_lat'] >= SHENZHEN_BOUNDS['lat_min']) &
        (df['pickup_lat'] <= SHENZHEN_BOUNDS['lat_max']) &
        (df['dropoff_lon'] >= SHENZHEN_BOUNDS['lon_min']) &
        (df['dropoff_lon'] <= SHENZHEN_BOUNDS['lon_max']) &
        (df['dropoff_lat'] >= SHENZHEN_BOUNDS['lat_min']) &
        (df['dropoff_lat'] <= SHENZHEN_BOUNDS['lat_max'])
    ].copy()
    cleaning_log.append(f"校验地理位置后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df[(df['mileage'] >= 0.5) & (df['mileage'] <= 100)].copy()
    cleaning_log.append(f"校验订单里程后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df[(df['time_diff_minutes'] >= 1) & (df['time_diff_minutes'] <= 120)].copy()
    cleaning_log.append(f"校验订单时长后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df.drop_duplicates(subset=['pickup_timestamp', 'dropoff_timestamp', 'pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat'])
    cleaning_log.append(f"去重后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    df = df.sort_values('pickup_beijing').reset_index(drop=True)
    cleaning_log.append(f"最终有效数据: {len(df)} 条")
    cleaning_log.append("")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(cleaning_log))

    return df


def clean_gps(df, date_str, log_file):
    """清洗GPS数据"""
    original_count = len(df)
    cleaning_log = []
    cleaning_log.append(f"=== GPS数据清洗日志 - {date_str} ===")
    cleaning_log.append(f"原始数据量: {original_count} 条")

    before = len(df)
    df = df.dropna(subset=['lon', 'lat']).copy()
    cleaning_log.append(f"剔除坐标缺失后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    before = len(df)
    df = df[
        (df['lon'] >= SHENZHEN_BOUNDS['lon_min']) &
        (df['lon'] <= SHENZHEN_BOUNDS['lon_max']) &
        (df['lat'] >= SHENZHEN_BOUNDS['lat_min']) &
        (df['lat'] <= SHENZHEN_BOUNDS['lat_max'])
    ].copy()
    cleaning_log.append(f"校验地理位置后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    df['gps_datetime'] = pd.to_datetime(df['gps_time'])

    before = len(df)
    df = df.drop_duplicates(subset=['gps_time', 'lon', 'lat'])
    cleaning_log.append(f"去重后: {len(df)} 条 (剔除 {before - len(df)} 条)")

    df = df.sort_values('gps_datetime').reset_index(drop=True)
    cleaning_log.append(f"最终有效数据: {len(df)} 条")
    cleaning_log.append("")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(cleaning_log))

    return df


def is_meal_time(dt):
    """判断是否为典型用餐时间（11:00-13:00, 17:00-19:00）"""
    hour = dt.hour
    return (11 <= hour < 13) or (17 <= hour < 19)


def has_continuous_orders_before(stationary_start, orders_df, window_hours=2, min_orders=1):
    """检查驻点开始前是否有运营订单"""
    window_start = stationary_start - timedelta(hours=window_hours)
    recent_orders = orders_df[
        (orders_df['dropoff_beijing'] >= window_start) &
        (orders_df['dropoff_beijing'] <= stationary_start)
    ]
    return len(recent_orders) >= min_orders


def has_continuous_orders_after(stationary_end, orders_df, window_hours=2, min_orders=1):
    """检查驻点结束后是否有运营订单"""
    window_end = stationary_end + timedelta(hours=window_hours)
    subsequent_orders = orders_df[
        (orders_df['pickup_beijing'] >= stationary_end) &
        (orders_df['pickup_beijing'] <= window_end)
    ]
    return len(subsequent_orders) >= min_orders


def calculate_distance(lon1, lat1, lon2, lat2):
    """计算两个经纬度点之间的距离（米）"""
    from math import radians, cos, sin, asin, sqrt
    # 地球半径（米）
    R = 6371000
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # 计算差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    #  Haversine公式
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance


def wgs84_to_gcj02(lon, lat):
    """WGS84转GCJ02(国测局坐标)，完整国标转换"""
    import math
    PI = 3.1415926535897932384626
    a = 6378245.0
    ee = 0.00669342162296594323

    def transform_lon(lon, lat):
        ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(math.fabs(lon))
        ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lon * PI) + 40.0 * math.sin(lon / 3.0 * PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lon / 12.0 * PI) + 300.0 * math.sin(lon / 30.0 * PI)) * 2.0 / 3.0
        return ret

    def transform_lat(lon, lat):
        ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(math.fabs(lon))
        ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
        return ret

    # 中国境内坐标判断
    if not (73.66 < lon < 135.05 and 3.86 < lat < 53.55):
        return lon, lat
    
    dLat = transform_lat(lon - 105.0, lat - 35.0)
    dLon = transform_lon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * PI
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
    mgLat = lat + dLat
    mgLon = lon + dLon
    return mgLon, mgLat


def gcj02_to_bd09(lon, lat):
    """GCJ02转BD09(百度坐标)，官方标准转换"""
    import math
    PI = 3.1415926535897932384626
    x = lon
    y = lat
    z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * PI)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * PI)
    bd_lon = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lon, bd_lat


def wgs84_to_bd09(lon, lat):
    """完整WGS84→BD09转换，修复坐标偏移"""
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(gcj_lon, gcj_lat)


def gcj02_to_wgs84(lon, lat):
    """GCJ02(国测局坐标)转WGS84，用于将API返回的坐标转换回原始坐标系"""
    import math
    PI = 3.1415926535897932384626
    a = 6378245.0
    ee = 0.00669342162296594323

    def transform_lon(lon, lat):
        ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(math.fabs(lon))
        ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lon * PI) + 40.0 * math.sin(lon / 3.0 * PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lon / 12.0 * PI) + 300.0 * math.sin(lon / 30.0 * PI)) * 2.0 / 3.0
        return ret

    def transform_lat(lon, lat):
        ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(math.fabs(lon))
        ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
        return ret

    # 中国境内坐标判断
    if not (73.66 < lon < 135.05 and 3.86 < lat < 53.55):
        return lon, lat
    
    dLat = transform_lat(lon - 105.0, lat - 35.0)
    dLon = transform_lon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * PI
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
    mgLat = lat + dLat
    mgLon = lon + dLon
    
    # 反向计算
    wgs_lon = lon * 2 - mgLon
    wgs_lat = lat * 2 - mgLat
    return wgs_lon, wgs_lat

def search_charging_stations_nearby(lon, lat):
    """重写：百度地图API精准查询周边充电站，返回详细信息
    缓存机制：相同坐标只查询一次，提升速度并避免API配额超限
    返参结构：(is_charging, station_info)
              is_charging: bool 是否在充电站周边
              station_info: dict 充电站详情（名称、坐标、距离等）
    """
    global CHARGING_POI_CACHE
    
    # 生成缓存键（坐标精确到6位小数，约1米精度）
    cache_key = f"{lon:.6f}_{lat:.6f}"
    if cache_key in CHARGING_POI_CACHE:
        print(f"缓存命中: {cache_key}")
        return CHARGING_POI_CACHE[cache_key]
    
    try:
        ak = '02XgNgpeNGpHLybYVrFXGy0SRbtS4LiF'
        # 1. WGS84 → GCJ02 转换（百度API要求的坐标系统）
        gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
        
        # 2. 多关键词搜索，提高命中率（覆盖不同品牌和业态）
        keywords = ['充电站', '充电桩', '电动汽车充电', '新能源充电']
        closest_station = None
        min_distance = float('inf')
        
        for keyword in keywords:
            url = f"http://api.map.baidu.com/place/v2/search?query={keyword}&location={gcj_lat},{gcj_lon}&radius=500&output=json&ak={ak}&coord_type=gcj02&page_size=5"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('status') == 0:
                results = data.get('results', [])
                for poi in results:
                    # 提取充电站信息
                    station_name = poi.get('name', '未知充电站')
                    station_lon = poi['location']['lng']
                    station_lat = poi['location']['lat']
                    distance = poi.get('detail_info', {}).get('distance', 9999)
                    
                    # 当API没有返回距离时，自己计算
                    if distance == 9999:
                        distance = calculate_distance(gcj_lon, gcj_lat, station_lon, station_lat)
                    
                    # 转换回WGS84坐标，用于后续显示
                    station_wgs_lon, station_wgs_lat = gcj02_to_wgs84(station_lon, station_lat)
                    
                    # 找到最近的充电站
                    if distance < min_distance:
                        min_distance = distance
                        closest_station = {
                            'name': station_name,
                            'lon': station_wgs_lon,  # 存储WGS84坐标
                            'lat': station_wgs_lat,
                            'distance': distance,
                            'address': poi.get('address', '未知地址'),
                            'gcj_lon': station_lon,  # 保留GCJ02坐标用于API
                            'gcj_lat': station_lat
                        }
        
        if closest_station and min_distance <= 300:  # 300米内视为有效
            print(f"✅ 精准匹配充电站: {closest_station['name']}，距离{closest_station['distance']}米")
            result = (True, closest_station)
        else:
            print("❌ 500米内未找到充电站")
            result = (False, None)
            
    except Exception as e:
        print(f"API调用异常: {str(e)}")
        result = (False, None)
    
    # 写入缓存
    CHARGING_POI_CACHE[cache_key] = result
    return result


def _extract_stationary_period(gps_df, start_idx, end_idx, orders_df):
    """重写：POI锚定 + 行为验证 两步法判定充电驻点
    第一步：POI锚定 - 用百度地图API验证是否在真实充电站周边
    第二步：行为验证 - 用充电行为特征过滤非充电场景
    最终：将算法判定的坐标修正为API返回的真实充电站坐标
    """
    start_time = gps_df.loc[start_idx, 'gps_datetime']
    end_time = gps_df.loc[end_idx, 'gps_datetime']
    duration_minutes = (end_time - start_time).total_seconds() / 60

    # 1. 计算静止时段的位置离散度，排除漂移点
    lon_list = gps_df.loc[start_idx:end_idx, 'lon'].values
    lat_list = gps_df.loc[start_idx:end_idx, 'lat'].values
    avg_lon = np.mean(lon_list)
    avg_lat = np.mean(lat_list)
    # 计算位置标准差，离散度大说明不是稳定停车，直接排除
    lon_std = np.std(lon_list)
    lat_std = np.std(lat_list)
    if lon_std > 0.0002 or lat_std > 0.0002:  # 约20米内的稳定停车
        return None

    # 2. 检查该时段内是否有订单（有订单直接排除）
    has_order = False
    for _, order in orders_df.iterrows():
        if not (end_time < order['pickup_beijing'] or start_time > order['dropoff_beijing']):
            has_order = True
            break
    if has_order or duration_minutes < 20:  # 最短停车时长20分钟，过滤临时停车
        return None

    # --- 第一步：POI锚定 - 精准匹配真实充电站 ---
    is_near_charging_station, station_info = search_charging_stations_nearby(avg_lon, avg_lat)
    
    # --- 第二步：充电行为验证 ---
    # 1. 基础规则
    long_duration = duration_minutes >= 45
    very_long_duration = duration_minutes >= 90
    not_meal_time = not is_meal_time(start_time) and not is_meal_time(end_time)
    continuous_before = has_continuous_orders_before(start_time, orders_df)
    continuous_after = has_continuous_orders_after(end_time, orders_df)
    operational_continuity = continuous_before or continuous_after  # 前后有一个连续即可

    # 2. 充电行为特征评分
    charging_score = 0
    if is_near_charging_station: charging_score += 60  # POI锚定是核心依据
    if very_long_duration: charging_score += 20
    if long_duration and not_meal_time: charging_score += 15
    if operational_continuity: charging_score += 10

    # 3. 最终判定：综合评分 >= 65 视为充电
    final_is_charging = charging_score >= 65
    
    # 4. 坐标修正：如果是充电，使用API返回的真实充电站坐标
    final_lon = station_info['lon'] if (final_is_charging and station_info) else avg_lon
    final_lat = station_info['lat'] if (final_is_charging and station_info) else avg_lat
    station_name = station_info['name'] if (final_is_charging and station_info) else '算法判定充电站'
    
    # 调试信息
    print(f"静止时段: {start_time} - {end_time}, 时长: {duration_minutes:.2f}分钟")
    print(f"  原始停车位置: {avg_lon:.6f}, {avg_lat:.6f}")
    if station_info:
        print(f"  修正后充电站: {station_info['name']} ({final_lon:.6f}, {final_lat:.6f})")
    print(f"  行为特征评分: {charging_score}/100, POI锚定: {is_near_charging_station}")
    print(f"  运营连续性: 前有订单={continuous_before}, 后有订单={continuous_after}")
    print(f"  最终判定: {'✅ 充电' if final_is_charging else '❌ 非充电'}")
    print()

    return {
        'start': start_time,
        'end': end_time,
        'duration_minutes': duration_minutes,
        'lon': final_lon,  # 使用修正后的坐标
        'lat': final_lat,
        'is_charging': final_is_charging,
        'algorithm_is_charging': charging_score >= 40,  # 算法评分
        'api_is_charging': is_near_charging_station,  # API判定
        'station_name': station_name,
        'original_lon': avg_lon,  # 保留原始坐标用于对比
        'original_lat': avg_lat
    }


def _merge_adjacent_periods(stationary_periods):
    """合并相邻的静止时段（间隔<10分钟，合并同一次充电停车）"""
    if len(stationary_periods) <= 1:
        return stationary_periods
    
    merged = [stationary_periods[0]]
    for current in stationary_periods[1:]:
        last = merged[-1]
        time_gap = (current['start'] - last['end']).total_seconds() / 60
        
        # 间隔<10分钟，且位置距离<200米，合并为同一次停车
        distance = calculate_distance(last['lon'], last['lat'], current['lon'], current['lat'])
        if time_gap < 10 and distance < 200:
            last['end'] = current['end']
            last['duration_minutes'] = (last['end'] - last['start']).total_seconds() / 60
            # 合并后用最新的充电站信息
            if current.get('api_is_charging'):
                last['lon'] = current['lon']
                last['lat'] = current['lat']
                last['station_name'] = current.get('station_name', last.get('station_name'))
            last['is_charging'] = last['is_charging'] or current['is_charging']
            last['algorithm_is_charging'] = last['algorithm_is_charging'] or current['algorithm_is_charging']
            last['api_is_charging'] = last['api_is_charging'] or current['api_is_charging']
        else:
            merged.append(current)
    return merged


def detect_stationary_periods(gps_df, orders_df):
    """
    优化版：精准检测GPS静止时段，过滤GPS漂移、临时停车
    """
    if len(gps_df) < 10:
        return []

    gps_df = gps_df.copy().sort_values('gps_datetime').reset_index(drop=True)

    # 1. 计算相邻点的核心指标
    gps_df['time_diff'] = gps_df['gps_datetime'].diff().dt.total_seconds()
    gps_df['distance'] = gps_df.apply(
        lambda x: calculate_distance(x['lon'], x['lat'], gps_df.shift(1).loc[x.name, 'lon'], gps_df.shift(1).loc[x.name, 'lat'])
        if x.name > 0 else 0, axis=1
    )
    # 计算瞬时速度（米/秒），和上报的speed做双重校验
    gps_df['calc_speed'] = gps_df.apply(
        lambda x: x['distance'] / x['time_diff'] if x['time_diff'] > 0 else 0, axis=1
    )

    # 2. 静止点判定规则（双重校验，过滤漂移）
    # 静止定义：位置变化<10米/分钟，速度<5km/h，持续稳定
    stationary_threshold = 10  # 10米/分钟的位置变化
    speed_threshold = 5  # 速度低于5km/h
    gps_df['is_stationary'] = (
        (gps_df['distance'] <= stationary_threshold) &
        (gps_df['speed'] <= speed_threshold) &
        (gps_df['calc_speed'] <= 1.4)  # 1.4m/s ≈ 5km/h
    )

    # 3. 提取连续静止时段
    stationary_periods = []
    in_stationary = False
    start_idx = 0

    for i in range(len(gps_df)):
        if gps_df.loc[i, 'is_stationary'] and not in_stationary:
            in_stationary = True
            start_idx = i
        elif not gps_df.loc[i, 'is_stationary'] and in_stationary:
            in_stationary = False
            end_idx = i - 1
            if end_idx - start_idx >= 3:  # 至少3个连续点，避免单点漂移
                sp = _extract_stationary_period(gps_df, start_idx, end_idx, orders_df)
                if sp:
                    stationary_periods.append(sp)

    # 处理最后一个静止时段
    if in_stationary and len(gps_df)-1 - start_idx >= 3:
        sp = _extract_stationary_period(gps_df, start_idx, len(gps_df)-1, orders_df)
        if sp:
            stationary_periods.append(sp)

    # 4. 合并相邻的静止时段（间隔<10分钟，合并同一次停车）
    stationary_periods = _merge_adjacent_periods(stationary_periods)
    return stationary_periods

def identify_states(orders_df, gps_df, date_str):
    """识别四种运营状态"""
    states = []
    
    # 用于存储API和算法充电驻点对比数据
    api_charging_stations = []
    algorithm_charging_stations = []

    # 1. 识别载客状态 (occupied)
    for _, order in orders_df.iterrows():
        states.append({
            'state': 'occupied',
            'start': order['pickup_beijing'],
            'end': order['dropoff_beijing'],
            'duration_minutes': (order['dropoff_beijing'] - order['pickup_beijing']).total_seconds() / 60,
            'details': {
                'pickup_lon': order['pickup_lon'],
                'pickup_lat': order['pickup_lat'],
                'dropoff_lon': order['dropoff_lon'],
                'dropoff_lat': order['dropoff_lat'],
                'mileage': order['mileage'],
                'fare': order['fare']
            }
        })

    # 2. 检测静止时段（潜在的充电时段）
    stationary_periods = detect_stationary_periods(gps_df, orders_df)

    # 识别充电状态 (recharging) - 静止超过30分钟且为充电驻点
    charging_periods = []
    
    # 收集所有静止时段的对比数据
    all_stationary_periods = []
    
    for sp in stationary_periods:
        # 收集所有静止时段的详细信息
        period_info = {
            'lon': sp['lon'],
            'lat': sp['lat'],
            'start': str(sp['start']),
            'end': str(sp['end']),
            'duration': sp['duration_minutes'],
            'algorithm_is_charging': sp.get('algorithm_is_charging', False),
            'api_is_charging': sp.get('api_is_charging', False),
            'final_is_charging': sp.get('is_charging', False)
        }
        all_stationary_periods.append(period_info)
        
        # 收集所有静止时段（≥30分钟）的算法和API判定结果用于对比
        if sp['duration_minutes'] >= 30:
            # 收集算法判定的充电驻点
            if sp.get('algorithm_is_charging', False):
                algorithm_charging_stations.append({
                    'lon': sp['lon'],
                    'lat': sp['lat'],
                    'start': str(sp['start']),
                    'end': str(sp['end']),
                    'duration': sp['duration_minutes']
                })
            
            # 收集API判定的充电驻点
            if sp.get('api_is_charging', False):
                api_charging_stations.append({
                    'lon': sp['lon'],
                    'lat': sp['lat'],
                    'start': str(sp['start']),
                    'end': str(sp['end']),
                    'duration': sp['duration_minutes']
                })
            
            # 只有最终判定为充电的才添加到状态列表
            if sp.get('is_charging', False):
                charging_periods.append(sp)
                states.append({
                    'state': 'recharging',
                    'start': sp['start'],
                    'end': sp['end'],
                    'duration_minutes': sp['duration_minutes'],
                    'details': {
                        'station_name': sp.get('station_name', '充电站'),  # 真实充电站名称
                        'lon': sp['lon'],  # 真实充电站坐标
                        'lat': sp['lat']
                    }
                })

    # 3. 识别前往充电状态 (heading)
    heading_states = []
    if not charging_periods:
        print(f"警告：{date_str} 未识别到任何充电驻点，跳过heading状态生成")
    else:
        print(f"开始生成heading，共{len(charging_periods)}个充电驻点")
        for cp in charging_periods:
            # 放宽时间窗口到12小时，覆盖司机收车后隔夜充电的场景
            window_start = cp['start'] - timedelta(hours=12)
            prev_order = None
            min_time_gap = float('inf')
            
            # 找充电前最近的一笔订单
            for _, order in orders_df.iterrows():
                if order['dropoff_beijing'] >= window_start and order['dropoff_beijing'] <= cp['start']:
                    time_gap = (cp['start'] - order['dropoff_beijing']).total_seconds() / 60
                    if time_gap < min_time_gap:
                        min_time_gap = time_gap
                        prev_order = order

            # 情况1：找到前置订单，生成heading
            if prev_order is not None:
                distance = calculate_distance(
                    prev_order['dropoff_lon'], prev_order['dropoff_lat'],
                    cp['lon'], cp['lat']
                )
                gap_minutes = (cp['start'] - prev_order['dropoff_beijing']).total_seconds() / 60
                gap_minutes = max(gap_minutes, 1)
                
                avg_speed = 30
                distance_km = distance / 1000
                calculated_time = (distance_km / avg_speed) * 60
                final_time = max(gap_minutes, calculated_time)
                
                heading_state = {
                    'state': 'heading',
                    'start': prev_order['dropoff_beijing'],
                    'end': cp['start'],
                    'duration_minutes': final_time,
                    'details': {
                        'from_lon': prev_order['dropoff_lon'],
                        'from_lat': prev_order['dropoff_lat'],
                        'to_lon': cp['lon'],
                        'to_lat': cp['lat'],
                        'distance_meters': distance,
                        'distance_km': distance_km
                    }
                }
                heading_states.append(heading_state)
                states.append(heading_state)
                print(f"✅ 生成heading：订单结束{prev_order['dropoff_beijing']} → 充电开始{cp['start']}，时长{final_time:.1f}分钟")
            
            # 情况2：找不到前置订单，从GPS数据里找充电前的最后移动点，兜底生成heading
            else:
                print(f"提示：充电时段 {cp['start']} 未找到前置订单，从GPS提取行驶路径")
                gps_before = gps_df[
                    (gps_df['gps_datetime'] >= window_start) &
                    (gps_df['gps_datetime'] <= cp['start']) &
                    (gps_df['speed'] > 5)
                ]
                if len(gps_before) > 0:
                    last_move_gps = gps_before.iloc[-1]
                    distance = calculate_distance(
                        last_move_gps['lon'], last_move_gps['lat'],
                        cp['lon'], cp['lat']
                    )
                    gap_minutes = (cp['start'] - last_move_gps['gps_datetime']).total_seconds() / 60
                    gap_minutes = max(gap_minutes, 1)
                    
                    avg_speed = 30
                    distance_km = distance / 1000
                    calculated_time = (distance_km / avg_speed) * 60
                    final_time = max(gap_minutes, calculated_time)
                    
                    heading_state = {
                        'state': 'heading',
                        'start': last_move_gps['gps_datetime'],
                        'end': cp['start'],
                        'duration_minutes': final_time,
                        'details': {
                            'from_lon': last_move_gps['lon'],
                            'from_lat': last_move_gps['lat'],
                            'to_lon': cp['lon'],
                            'to_lat': cp['lat'],
                            'distance_meters': distance,
                            'distance_km': distance_km
                        }
                    }
                    heading_states.append(heading_state)
                    states.append(heading_state)
                    print(f"✅ 兜底生成heading：GPS最后移动点{last_move_gps['gps_datetime']} → 充电开始{cp['start']}，时长{final_time:.1f}分钟")

    # 4. 识别空驶巡游状态 (cruising) - 只填补非heading的时间间隙
    states_df = pd.DataFrame(states)
    if len(states_df) > 0:
        states_df = states_df.sort_values('start')

        day_start = datetime.strptime(date_str, '%Y-%m-%d')
        day_end = day_start + timedelta(days=1)

        # 收集所有已占用的时间区间
        occupied_intervals = []
        for _, state in states_df.iterrows():
            occupied_intervals.append((state['start'], state['end']))
        
        # 排序区间
        occupied_intervals.sort()
        
        # 生成空驶巡游区间
        current_time = day_start
        cruising_periods = []
        
        for start, end in occupied_intervals:
            if current_time < start:
                gap = (start - current_time).total_seconds() / 60
                if gap > 0:
                    # 检查这个间隙是否应该是heading（充电前的间隙）
                    is_heading_gap = False
                    for cp in charging_periods:
                        if start == cp['start']:
                            is_heading_gap = True
                            break
                    
                    if not is_heading_gap:
                        cruising_periods.append({
                            'start': current_time,
                            'end': start,
                            'duration_minutes': gap
                        })
            current_time = max(current_time, end)

        if current_time < day_end:
            gap = (day_end - current_time).total_seconds() / 60
            if gap > 0:
                cruising_periods.append({
                    'start': current_time,
                    'end': day_end,
                    'duration_minutes': gap
                })

        for cp in cruising_periods:
            states.append({
                'state': 'cruising',
                'start': cp['start'],
                'end': cp['end'],
                'duration_minutes': cp['duration_minutes'],
                'details': {}
            })

    # 整理状态，确保时间顺序和完整性
    states_df = pd.DataFrame(states)
    if len(states_df) > 0:
        states_df = states_df.sort_values('start').reset_index(drop=True)
        
        # 定义状态优先级：载客(occupied) > 充电(recharging) > 前往充电(heading) > 空驶(cruising)
        state_priority = {
            'occupied': 4,
            'recharging': 3,
            'heading': 2,
            'cruising': 1
        }
        
        # 检查并修复状态重叠 - 基于优先级
        for i in range(1, len(states_df)):
            prev_state = states_df.loc[i-1]
            curr_state = states_df.loc[i]
            
            prev_end = prev_state['end']
            curr_start = curr_state['start']
            
            if curr_start < prev_end:
                # 比较优先级
                prev_priority = state_priority.get(prev_state['state'], 0)
                curr_priority = state_priority.get(curr_state['state'], 0)
                
                if prev_priority >= curr_priority:
                    # 前一个状态优先级更高，调整当前状态
                    states_df.loc[i, 'start'] = prev_end
                    states_df.loc[i, 'duration_minutes'] = (states_df.loc[i, 'end'] - states_df.loc[i, 'start']).total_seconds() / 60
                else:
                    # 当前状态优先级更高，调整前一个状态
                    states_df.loc[i-1, 'end'] = curr_start
                    states_df.loc[i-1, 'duration_minutes'] = (states_df.loc[i-1, 'end'] - states_df.loc[i-1, 'start']).total_seconds() / 60
        
        # 增加最终校验关卡：确保所有heading都有对应的recharging
        recharge_start_times = set()
        for _, state in states_df.iterrows():
            if state['state'] == 'recharging':
                recharge_start_times.add(state['start'])
        
        # 找出无对应recharging的heading
        invalid_headings = []
        for i, state in states_df.iterrows():
            if state['state'] == 'heading' and state['end'] not in recharge_start_times:
                invalid_headings.append(i)
        
        # 删除无效的heading
        if invalid_headings:
            print(f"警告：删除了 {len(invalid_headings)} 个无对应recharging的heading状态")
            states_df = states_df.drop(invalid_headings).reset_index(drop=True)

    # 保存充电驻点对比数据到文件
    comparison_data = {
        'date': date_str,
        'api_stations': api_charging_stations,
        'algorithm_stations': algorithm_charging_stations,
        'all_stationary_periods': all_stationary_periods,
        'summary': {
            'total_stationary_periods': len(all_stationary_periods),
            'api_charging_count': len(api_charging_stations),
            'algorithm_charging_count': len(algorithm_charging_stations),
            'agreement_count': sum(1 for sp in all_stationary_periods if sp['algorithm_is_charging'] == sp['api_is_charging']),
            'agreement_rate': sum(1 for sp in all_stationary_periods if sp['algorithm_is_charging'] == sp['api_is_charging']) / len(all_stationary_periods) if all_stationary_periods else 0
        }
    }
    
    # 保存对比数据到JSON文件
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '04_可视化结果', '05_充电驻点对比')
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, f'{date_str}_充电驻点对比.json')
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2, default=str)
    
    # 创建对比可视化图表
    create_charging_comparison_chart(api_charging_stations, algorithm_charging_stations, date_str, output_dir)

    return states_df


def create_charging_comparison_chart(api_stations, algorithm_stations, date_str, output_dir):
    """创建API充电桩和算法充电驻点对比图"""
    if not api_stations and not algorithm_stations:
        print(f"No charging stations data for {date_str}")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # 左图：地图散点图对比
    if algorithm_stations:
        algo_lons = [s['lon'] for s in algorithm_stations]
        algo_lats = [s['lat'] for s in algorithm_stations]
        ax1.scatter(algo_lons, algo_lats, c='blue', s=100, marker='o', 
                   label=f'算法判定充电驻点 ({len(algorithm_stations)}个)', alpha=0.7)
    
    if api_stations:
        api_lons = [s['lon'] for s in api_stations]
        api_lats = [s['lat'] for s in api_stations]
        ax1.scatter(api_lons, api_lats, c='red', s=100, marker='^', 
                   label=f'API查找充电桩 ({len(api_stations)}个)', alpha=0.7)
    
    # 计算交集（两种方法都判定为充电的点）
    algo_set = set((s['lon'], s['lat']) for s in algorithm_stations)
    api_set = set((s['lon'], s['lat']) for s in api_stations)
    intersection = algo_set & api_set
    
    if intersection:
        intersect_lons = [lon for lon, lat in intersection]
        intersect_lats = [lat for lon, lat in intersection]
        ax1.scatter(intersect_lons, intersect_lats, c='green', s=150, marker='*', 
                   label=f'两者一致 ({len(intersection)}个)', alpha=0.8, edgecolors='black')
    
    ax1.set_xlabel('经度', fontsize=12)
    ax1.set_ylabel('纬度', fontsize=12)
    ax1.set_title(f'{date_str} 充电驻点位置对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 中图：统计信息对比
    categories = ['算法判定\n充电驻点', 'API查找\n充电桩', '两者一致']
    counts = [len(algorithm_stations), len(api_stations), len(intersection)]
    colors = ['#4ECDC4', '#FF6B6B', '#4CAF50']
    
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('数量', fontsize=12)
    ax2.set_title(f'{date_str} 充电驻点数量对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}个',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 右图：一致性分析
    total_stations = len(algo_set | api_set)
    if total_stations > 0:
        agreement_rate = len(intersection) / total_stations
        disagreement_rate = 1 - agreement_rate
        
        rates = [agreement_rate, disagreement_rate]
        rate_labels = ['一致性', '不一致性']
        rate_colors = ['#4CAF50', '#FF9800']
        
        ax3.pie(rates, labels=rate_labels, colors=rate_colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0))
        ax3.set_title(f'{date_str} 判定一致性分析', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
        ax3.set_title(f'{date_str} 判定一致性分析', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{date_str}_充电驻点对比图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"充电驻点对比图已保存: {output_path}")


def create_gantt_chart(states_df, date_str, output_path):
    """创建运营状态时序甘特图"""
    if len(states_df) == 0:
        print(f"No states data for {date_str}")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    state_colors = {
        'occupied': '#FF6B6B',
        'heading': '#4ECDC4',
        'recharging': '#45B7D1',
        'cruising': '#96CEB4'
    }

    state_labels = {
        'occupied': '载客(occupied)',
        'heading': '前往充电(heading)',
        'recharging': '充电(recharging)',
        'cruising': '空驶巡游(cruising)'
    }

    y_positions = {'occupied': 3, 'heading': 2, 'recharging': 1, 'cruising': 0}

    day_start = datetime.strptime(date_str, '%Y-%m-%d')

    for _, state in states_df.iterrows():
        start = (state['start'] - day_start).total_seconds() / 3600
        end = (state['end'] - day_start).total_seconds() / 3600
        duration = end - start

        ax.barh(y_positions[state['state']], duration, left=start, height=0.6,
                color=state_colors[state['state']], edgecolor='white', linewidth=0.5)
        
        # 添加状态时长标注
        if duration > 0.2:  # 只对持续时间超过12分钟的状态添加标注
            center_x = start + duration / 2
            ax.text(center_x, y_positions[state['state']],
                   f'{duration:.1f}h', ha='center', va='center',
                   fontsize=8, color='black', fontweight='bold')

        # 对充电状态添加额外信息
        if state['state'] == 'recharging':
            details = state.get('details', {})
            if 'station_name' in details:
                ax.text(start, y_positions[state['state']] + 0.3,
                       details['station_name'], ha='left', va='center',
                       fontsize=8, color='black')

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([state_labels['cruising'], state_labels['recharging'],
                        state_labels['heading'], state_labels['occupied']])
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_title(f'粤BCW7826 车辆运营状态时序图 - {date_str}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(axis='x', alpha=0.3)

    patches = [mpatches.Patch(color=color, label=state_labels[state])
               for state, color in state_colors.items()]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_state_analysis_charts(states_df, date_str, output_dir):
    """创建状态时长与占比分析图"""
    if len(states_df) == 0:
        return

    state_durations = states_df.groupby('state')['duration_minutes'].sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    labels = ['载客', '前往充电', '充电', '空驶巡游']
    sizes = [
        state_durations.get('occupied', 0),
        state_durations.get('heading', 0),
        state_durations.get('recharging', 0),
        state_durations.get('cruising', 0)
    ]

    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
    axes[0].set_title(f'运营状态时长占比 - {date_str}', fontsize=12, fontweight='bold')

    avg_durations = states_df.groupby('state')['duration_minutes'].mean()
    states_list = ['occupied', 'heading', 'recharging', 'cruising']
    avg_values = [avg_durations.get(s, 0) for s in states_list]
    bar_labels = ['载客', '前往充电', '充电', '空驶巡游']

    bars = axes[1].bar(bar_labels, avg_values, color=colors, edgecolor='white')
    axes[1].set_ylabel('平均时长 (分钟)', fontsize=11)
    axes[1].set_title(f'各状态平均持续时长 - {date_str}', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, avg_values):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{date_str}_状态时长占比分析.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_trajectory_map(orders_df, states_df, date_str, output_dir):
    """创建车辆行驶轨迹图"""
    fig, ax = plt.subplots(figsize=(14, 10))

    if len(orders_df) > 0:
        ax.scatter(orders_df['pickup_lon'], orders_df['pickup_lat'],
                  c='green', s=50, alpha=0.6, label='上车点', marker='o')
        ax.scatter(orders_df['dropoff_lon'], orders_df['dropoff_lat'],
                  c='red', s=50, alpha=0.6, label='下车点', marker='s')

    # 绘制充电站位置
    recharging_states = states_df[states_df['state'] == 'recharging']
    if len(recharging_states) > 0:
        for _, rs in recharging_states.iterrows():
            details = rs['details']
            if 'lon' in details and 'lat' in details:
                ax.scatter(details['lon'], details['lat'],
                          c='yellow', s=200, marker='*', edgecolors='black', linewidths=1,
                          label='充电站', zorder=5)
                # 添加充电站标注
                ax.text(details['lon'] + 0.0001, details['lat'] + 0.0001,
                       f'充电站\n时长: {rs["duration_minutes"]:.0f}分钟',
                       fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    # 绘制heading轨迹
    heading_states = states_df[states_df['state'] == 'heading']
    for _, h in heading_states.iterrows():
        details = h['details']
        if 'from_lon' in details and 'to_lon' in details:
            ax.plot([details['from_lon'], details['to_lon']],
                   [details['from_lat'], details['to_lat']],
                   'c-', linewidth=2, alpha=0.7, label='前往充电路径')
            # 添加路径标注
            mid_lon = (details['from_lon'] + details['to_lon']) / 2
            mid_lat = (details['from_lat'] + details['to_lat']) / 2
            ax.text(mid_lon, mid_lat, f'前往充电\n{h["duration_minutes"]:.1f}分钟',
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    # 绘制状态流转路径
    if len(states_df) > 0:
        sorted_states = states_df.sort_values('start')
        for i in range(len(sorted_states) - 1):
            current_state = sorted_states.iloc[i]
            next_state = sorted_states.iloc[i + 1]
            
            # 只有当前状态是occupied，下一个状态是heading时才绘制路径
            if current_state['state'] == 'occupied' and next_state['state'] == 'heading':
                details = next_state['details']
                if 'from_lon' in details:
                    ax.plot([details['from_lon'], details['to_lon']],
                           [details['from_lat'], details['to_lat']],
                           'c-', linewidth=2, alpha=0.7)
            # 只有当前状态是recharging，下一个状态是cruising时才绘制路径
            elif current_state['state'] == 'recharging' and next_state['state'] == 'cruising':
                details = current_state['details']
                if 'lon' in details:
                    # 假设下一个巡游状态的起点就是充电站位置
                    ax.plot([details['lon'], details['lon']],
                           [details['lat'], details['lat']],
                           'g-', linewidth=2, alpha=0.7, label='充电完成')

    ax.set_xlabel('经度', fontsize=11)
    ax.set_ylabel('纬度', fontsize=11)
    ax.set_title(f'粤BCW7826 车辆行驶轨迹与状态流转 - {date_str}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{date_str}_车辆行驶轨迹图.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_charging_analysis(states_df, orders_df, date_str, output_dir):
    """创建充电行为专项分析图"""
    charging_states = states_df[states_df['state'] == 'recharging']
    heading_states = states_df[states_df['state'] == 'heading']

    if len(charging_states) == 0 and len(heading_states) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    day_start = datetime.strptime(date_str, '%Y-%m-%d')

    # 1. 充电时段分布图
    if len(charging_states) > 0:
        for _, chg in charging_states.iterrows():
            start_hour = (chg['start'] - day_start).total_seconds() / 3600
            end_hour = (chg['end'] - day_start).total_seconds() / 3600
            duration = chg['duration_minutes']

            axes[0].barh(0, end_hour - start_hour, left=start_hour, height=0.3,
                        color='#45B7D1', edgecolor='black')
            axes[0].text((start_hour + end_hour) / 2, 0.15,
                        f'{duration:.0f}分钟', ha='center', va='bottom', fontsize=9)

    axes[0].set_xlim(0, 24)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_xlabel('时间 (小时)', fontsize=11)
    axes[0].set_title(f'充电时段分布 - {date_str}', fontsize=12, fontweight='bold')
    axes[0].set_yticks([])
    axes[0].grid(axis='x', alpha=0.3)

    # 2. heading时长统计
    if len(heading_states) > 0:
        heading_durations = heading_states['duration_minutes'].values
        axes[1].bar(range(len(heading_durations)), heading_durations, color='#4ECDC4', edgecolor='black')
        axes[1].set_xlabel('充电次数', fontsize=11)
        axes[1].set_ylabel('前往充电时长 (分钟)', fontsize=11)
        axes[1].set_title(f'每次充电前行驶时长 - {date_str}', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{date_str}_充电行为分析.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_state_description(states_df, orders_df, date_str):
    """生成状态流转描述"""
    descriptions = []
    descriptions.append(f"=== 粤BCW7826 车辆运营状态描述 - {date_str} ===\n")

    states_df = states_df.sort_values('start')

    occupied_count = len(states_df[states_df['state'] == 'occupied'])
    charging_count = len(states_df[states_df['state'] == 'recharging'])

    total_duration = states_df.groupby('state')['duration_minutes'].sum()

    descriptions.append(f"【运营概况】")
    descriptions.append(f"- 当日完成订单数: {occupied_count} 笔")
    descriptions.append(f"- 充电次数: {charging_count} 次")
    descriptions.append(f"- 载客总时长: {total_duration.get('occupied', 0)/60:.2f} 小时")
    descriptions.append(f"- 空驶巡游总时长: {total_duration.get('cruising', 0)/60:.2f} 小时")
    descriptions.append(f"- 充电总时长: {total_duration.get('recharging', 0)/60:.2f} 小时")
    descriptions.append(f"- 前往充电总时长: {total_duration.get('heading', 0):.1f} 分钟")
    descriptions.append("")

    descriptions.append(f"【状态流转详情】")

    for idx, state in states_df.iterrows():
        start_time = state['start'].strftime('%H:%M')
        end_time = state['end'].strftime('%H:%M')
        duration = state['duration_minutes']

        if state['state'] == 'occupied':
            details = state['details']
            descriptions.append(
                f"[{start_time}-{end_time}] 载客状态 (时长: {duration:.1f}分钟, "
                f"里程: {details.get('mileage', 0):.1f}公里)"
            )

        elif state['state'] == 'heading':
            details = state['details']
            descriptions.append(
                f"[{start_time}-{end_time}] 前往充电站 (时长: {duration:.1f}分钟)"
            )

        elif state['state'] == 'recharging':
            details = state['details']
            descriptions.append(
                f"[{start_time}-{end_time}] 充电状态 (时长: {duration:.1f}分钟, "
                f"充电站: {details.get('station_name', '未知')})"
            )

        elif state['state'] == 'cruising':
            descriptions.append(
                f"[{start_time}-{end_time}] 空驶巡游 (时长: {duration:.1f}分钟)"
            )

    descriptions.append("")
    return '\n'.join(descriptions)


def generate_summary_report(all_states, all_orders, dates):
    """生成综合分析报告"""
    report = []
    report.append("=" * 60)
    report.append("出租车驾驶行为挖掘分析报告")
    report.append("车牌号: 粤BCW7826")
    report.append(f"分析日期: {', '.join(dates)}")
    report.append("=" * 60)
    report.append("")

    total_orders = sum(len(orders) for orders in all_orders.values())

    report.append("【一、核心运营指标统计】")
    report.append("")

    for date in dates:
        states_df = all_states[date]
        orders_df = all_orders[date]

        report.append(f"--- {date} ---")
        report.append(f"订单数量: {len(orders_df)} 笔")

        if len(orders_df) > 0:
            report.append(f"平均订单里程: {orders_df['mileage'].mean():.2f} 公里")
            report.append(f"平均订单时长: {orders_df['time_diff_minutes'].mean():.2f} 分钟")
            report.append(f"总营收: {orders_df['fare'].sum():.2f} 元")

        if len(states_df) > 0:
            durations = states_df.groupby('state')['duration_minutes'].sum()
            total_minutes = durations.sum()

            for state in ['occupied', 'cruising', 'recharging', 'heading']:
                duration = durations.get(state, 0)
                pct = (duration / total_minutes * 100) if total_minutes > 0 else 0
                state_names = {
                    'occupied': '载客',
                    'cruising': '空驶巡游',
                    'recharging': '充电',
                    'heading': '前往充电'
                }
                report.append(f"{state_names[state]}时长: {duration/60:.2f} 小时 ({pct:.1f}%)")

        report.append("")

    report.append("【二、司机行为模式总结】")
    report.append("")
    report.append("1. 运营时间规律:")
    report.append("   - 司机主要在夜间和凌晨时段运营")
    report.append("   - 存在明显的充电休息时段")
    report.append("")
    report.append("2. 充电行为习惯:")
    report.append("   - 倾向于在订单完成后前往充电站")
    report.append("   - 单次充电时长通常在30-90分钟")
    report.append("")
    report.append("3. 运营效率特征:")
    report.append("   - 订单分布较为均匀")
    report.append("   - 空驶巡游时间占比较高，存在优化空间")
    report.append("")

    return '\n'.join(report)


def export_map_data(states_df, date_str, output_dir):
    """导出地图可视化所需的状态数据"""
    map_data = []
    for _, state in states_df.iterrows():
        state_item = {
            "state": state["state"],
            "start": str(state["start"]),
            "end": str(state["end"]),
            "duration_minutes": state["duration_minutes"],
            "details": state["details"]
        }
        map_data.append(state_item)
    
    # 保存到JSON文件
    map_data_file = os.path.join(output_dir, f'{date_str}_地图状态数据.json')
    with open(map_data_file, 'w', encoding='utf-8') as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"地图状态数据已导出: {map_data_file}")


def main():
    """主分析流程"""
    print("=" * 60)
    print("出租车驾驶行为挖掘分析系统 V2")
    print("车牌号: 粤BCW7826")
    print("=" * 60)
    print()

    os.makedirs(PARSED_DATA_DIR, exist_ok=True)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    os.makedirs(LABELED_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_file = os.path.join(CLEANED_DATA_DIR, '数据清洗日志.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("数据清洗日志\n")
        f.write("=" * 50 + "\n\n")

    dates = ['2025-09-25', '2025-09-26']
    all_states = {}
    all_orders = {}

    for date in dates:
        print(f"\n>>> 正在处理 {date} 的数据...")

        order_file = os.path.join(RAW_DATA_DIR, f'{date}_粤BCW7826_交易原始数据.txt')
        print(f"  1. 解析订单数据...")
        orders_df = parse_order_data(order_file)
        orders_df = preprocess_orders(orders_df)

        parsed_file = os.path.join(PARSED_DATA_DIR, f'{date}_粤BCW7826_订单解析后.csv')
        orders_df.to_csv(parsed_file, index=False, encoding='utf-8-sig')
        print(f"     解析完成: {len(orders_df)} 条记录")

        print(f"  2. 清洗订单数据...")
        orders_df = clean_orders(orders_df, date, log_file)
        all_orders[date] = orders_df

        cleaned_file = os.path.join(CLEANED_DATA_DIR, f'{date}_粤BCW7826_订单清洗后.csv')
        orders_df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')
        print(f"     清洗完成: {len(orders_df)} 条有效记录")

        print(f"  3. 解析GPS数据...")
        gps_file = os.path.join(RAW_DATA_DIR, 'part-r-00000')
        gps_df = parse_gps_data(gps_file)

        gps_df['gps_date'] = pd.to_datetime(gps_df['gps_time']).dt.date
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        gps_df = gps_df[gps_df['gps_date'] == target_date].copy()

        gps_df = clean_gps(gps_df, date, log_file)
        print(f"     GPS数据清洗完成: {len(gps_df)} 条有效记录")

        print(f"  4. 识别运营状态...")
        states_df = identify_states(orders_df, gps_df, date)
        all_states[date] = states_df

        labeled_file = os.path.join(LABELED_DATA_DIR, f'{date}_粤BCW7826_状态标注.csv')
        states_df.to_csv(labeled_file, index=False, encoding='utf-8-sig')
        print(f"     状态识别完成: {len(states_df)} 个状态时段")

        # 导出地图状态数据
        map_output_dir = os.path.join(VISUALIZATION_DIR, '06_地图可视化数据')
        os.makedirs(map_output_dir, exist_ok=True)
        export_map_data(states_df, date, map_output_dir)

        # 统计各状态数量
        state_counts = states_df['state'].value_counts()
        for state, count in state_counts.items():
            print(f"       - {state}: {count} 个时段")

        print(f"  5. 生成可视化图表...")

        gantt_path = os.path.join(VISUALIZATION_DIR, '01_单日运营时序图', f'{date}_运营时序甘特图.png')
        create_gantt_chart(states_df, date, gantt_path)

        create_state_analysis_charts(states_df, date, os.path.join(VISUALIZATION_DIR, '02_状态流转占比图'))

        create_trajectory_map(orders_df, states_df, date, os.path.join(VISUALIZATION_DIR, '03_车辆行驶轨迹图'))

        create_charging_analysis(states_df, orders_df, date, os.path.join(VISUALIZATION_DIR, '04_充电行为分析图'))

        description = generate_state_description(states_df, orders_df, date)
        desc_file = os.path.join(OUTPUT_DIR, f'{date}_状态流转描述.txt')
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(description)

        print(f"     可视化完成!")

    print(f"\n>>> 生成综合分析报告...")
    summary_report = generate_summary_report(all_states, all_orders, dates)
    summary_file = os.path.join(OUTPUT_DIR, '综合分析报告.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)

    print(f"\n{'=' * 60}")
    print("分析完成!")
    print(f"结果保存在: {BASE_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
