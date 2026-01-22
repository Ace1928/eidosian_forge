import sys
from math import trunc
from typing import (
class ChineseCNLocale(Locale):
    names = ['zh', 'zh-cn']
    past = '{0}前'
    future = '{0}后'
    timeframes = {'now': '刚才', 'second': '1秒', 'seconds': '{0}秒', 'minute': '1分钟', 'minutes': '{0}分钟', 'hour': '1小时', 'hours': '{0}小时', 'day': '1天', 'days': '{0}天', 'week': '1周', 'weeks': '{0}周', 'month': '1个月', 'months': '{0}个月', 'year': '1年', 'years': '{0}年'}
    month_names = ['', '一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']
    month_abbreviations = ['', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12']
    day_names = ['', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    day_abbreviations = ['', '一', '二', '三', '四', '五', '六', '日']