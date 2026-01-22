import sys
from math import trunc
from typing import (
class JapaneseLocale(Locale):
    names = ['ja', 'ja-jp']
    past = '{0}前'
    future = '{0}後'
    and_word = ''
    timeframes = {'now': '現在', 'second': '1秒', 'seconds': '{0}秒', 'minute': '1分', 'minutes': '{0}分', 'hour': '1時間', 'hours': '{0}時間', 'day': '1日', 'days': '{0}日', 'week': '1週間', 'weeks': '{0}週間', 'month': '1ヶ月', 'months': '{0}ヶ月', 'year': '1年', 'years': '{0}年'}
    month_names = ['', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    month_abbreviations = ['', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12']
    day_names = ['', '月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日', '日曜日']
    day_abbreviations = ['', '月', '火', '水', '木', '金', '土', '日']