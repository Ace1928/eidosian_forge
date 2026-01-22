import sys
from math import trunc
from typing import (
class SwahiliLocale(Locale):
    names = ['sw', 'sw-ke', 'sw-tz']
    past = '{0} iliyopita'
    future = 'muda wa {0}'
    and_word = 'na'
    timeframes = {'now': 'sasa hivi', 'second': 'sekunde', 'seconds': 'sekunde {0}', 'minute': 'dakika moja', 'minutes': 'dakika {0}', 'hour': 'saa moja', 'hours': 'saa {0}', 'day': 'siku moja', 'days': 'siku {0}', 'week': 'wiki moja', 'weeks': 'wiki {0}', 'month': 'mwezi moja', 'months': 'miezi {0}', 'year': 'mwaka moja', 'years': 'miaka {0}'}
    meridians = {'am': 'asu', 'pm': 'mch', 'AM': 'ASU', 'PM': 'MCH'}
    month_names = ['', 'Januari', 'Februari', 'Machi', 'Aprili', 'Mei', 'Juni', 'Julai', 'Agosti', 'Septemba', 'Oktoba', 'Novemba', 'Desemba']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mac', 'Apr', 'Mei', 'Jun', 'Jul', 'Ago', 'Sep', 'Okt', 'Nov', 'Des']
    day_names = ['', 'Jumatatu', 'Jumanne', 'Jumatano', 'Alhamisi', 'Ijumaa', 'Jumamosi', 'Jumapili']
    day_abbreviations = ['', 'Jumatatu', 'Jumanne', 'Jumatano', 'Alhamisi', 'Ijumaa', 'Jumamosi', 'Jumapili']