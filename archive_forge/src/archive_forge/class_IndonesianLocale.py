import sys
from math import trunc
from typing import (
class IndonesianLocale(Locale):
    names = ['id', 'id-id']
    past = '{0} yang lalu'
    future = 'dalam {0}'
    and_word = 'dan'
    timeframes = {'now': 'baru saja', 'second': '1 sebentar', 'seconds': '{0} detik', 'minute': '1 menit', 'minutes': '{0} menit', 'hour': '1 jam', 'hours': '{0} jam', 'day': '1 hari', 'days': '{0} hari', 'week': '1 minggu', 'weeks': '{0} minggu', 'month': '1 bulan', 'months': '{0} bulan', 'quarter': '1 kuartal', 'quarters': '{0} kuartal', 'year': '1 tahun', 'years': '{0} tahun'}
    meridians = {'am': '', 'pm': '', 'AM': '', 'PM': ''}
    month_names = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Ags', 'Sept', 'Okt', 'Nov', 'Des']
    day_names = ['', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    day_abbreviations = ['', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']