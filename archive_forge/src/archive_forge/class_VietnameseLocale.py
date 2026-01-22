import sys
from math import trunc
from typing import (
class VietnameseLocale(Locale):
    names = ['vi', 'vi-vn']
    past = '{0} trước'
    future = '{0} nữa'
    timeframes = {'now': 'hiện tại', 'second': 'một giây', 'seconds': '{0} giây', 'minute': 'một phút', 'minutes': '{0} phút', 'hour': 'một giờ', 'hours': '{0} giờ', 'day': 'một ngày', 'days': '{0} ngày', 'week': 'một tuần', 'weeks': '{0} tuần', 'month': 'một tháng', 'months': '{0} tháng', 'year': 'một năm', 'years': '{0} năm'}
    month_names = ['', 'Tháng Một', 'Tháng Hai', 'Tháng Ba', 'Tháng Tư', 'Tháng Năm', 'Tháng Sáu', 'Tháng Bảy', 'Tháng Tám', 'Tháng Chín', 'Tháng Mười', 'Tháng Mười Một', 'Tháng Mười Hai']
    month_abbreviations = ['', 'Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']
    day_names = ['', 'Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    day_abbreviations = ['', 'Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'CN']