import sys
from math import trunc
from typing import (
class EsperantoLocale(Locale):
    names = ['eo', 'eo-xx']
    past = 'antaŭ {0}'
    future = 'post {0}'
    timeframes = {'now': 'nun', 'second': 'sekundo', 'seconds': '{0} kelkaj sekundoj', 'minute': 'unu minuto', 'minutes': '{0} minutoj', 'hour': 'un horo', 'hours': '{0} horoj', 'day': 'unu tago', 'days': '{0} tagoj', 'month': 'unu monato', 'months': '{0} monatoj', 'year': 'unu jaro', 'years': '{0} jaroj'}
    month_names = ['', 'januaro', 'februaro', 'marto', 'aprilo', 'majo', 'junio', 'julio', 'aŭgusto', 'septembro', 'oktobro', 'novembro', 'decembro']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'maj', 'jun', 'jul', 'aŭg', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'lundo', 'mardo', 'merkredo', 'ĵaŭdo', 'vendredo', 'sabato', 'dimanĉo']
    day_abbreviations = ['', 'lun', 'mar', 'mer', 'ĵaŭ', 'ven', 'sab', 'dim']
    meridians = {'am': 'atm', 'pm': 'ptm', 'AM': 'ATM', 'PM': 'PTM'}
    ordinal_day_re = '((?P<value>[1-3]?[0-9](?=a))a)'

    def _ordinal_number(self, n: int) -> str:
        return f'{n}a'