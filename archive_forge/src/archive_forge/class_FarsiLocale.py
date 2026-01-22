import sys
from math import trunc
from typing import (
class FarsiLocale(Locale):
    names = ['fa', 'fa-ir']
    past = '{0} قبل'
    future = 'در {0}'
    timeframes = {'now': 'اکنون', 'second': 'یک لحظه', 'seconds': '{0} ثانیه', 'minute': 'یک دقیقه', 'minutes': '{0} دقیقه', 'hour': 'یک ساعت', 'hours': '{0} ساعت', 'day': 'یک روز', 'days': '{0} روز', 'month': 'یک ماه', 'months': '{0} ماه', 'year': 'یک سال', 'years': '{0} سال'}
    meridians = {'am': 'قبل از ظهر', 'pm': 'بعد از ظهر', 'AM': 'قبل از ظهر', 'PM': 'بعد از ظهر'}
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_names = ['', 'دو شنبه', 'سه شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه', 'شنبه', 'یکشنبه']
    day_abbreviations = ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']