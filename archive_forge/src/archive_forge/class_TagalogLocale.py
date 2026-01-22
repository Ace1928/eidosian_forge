import sys
from math import trunc
from typing import (
class TagalogLocale(Locale):
    names = ['tl', 'tl-ph']
    past = 'nakaraang {0}'
    future = '{0} mula ngayon'
    timeframes = {'now': 'ngayon lang', 'second': 'isang segundo', 'seconds': '{0} segundo', 'minute': 'isang minuto', 'minutes': '{0} minuto', 'hour': 'isang oras', 'hours': '{0} oras', 'day': 'isang araw', 'days': '{0} araw', 'week': 'isang linggo', 'weeks': '{0} linggo', 'month': 'isang buwan', 'months': '{0} buwan', 'year': 'isang taon', 'years': '{0} taon'}
    month_names = ['', 'Enero', 'Pebrero', 'Marso', 'Abril', 'Mayo', 'Hunyo', 'Hulyo', 'Agosto', 'Setyembre', 'Oktubre', 'Nobyembre', 'Disyembre']
    month_abbreviations = ['', 'Ene', 'Peb', 'Mar', 'Abr', 'May', 'Hun', 'Hul', 'Ago', 'Set', 'Okt', 'Nob', 'Dis']
    day_names = ['', 'Lunes', 'Martes', 'Miyerkules', 'Huwebes', 'Biyernes', 'Sabado', 'Linggo']
    day_abbreviations = ['', 'Lun', 'Mar', 'Miy', 'Huw', 'Biy', 'Sab', 'Lin']
    meridians = {'am': 'nu', 'pm': 'nh', 'AM': 'ng umaga', 'PM': 'ng hapon'}

    def _ordinal_number(self, n: int) -> str:
        return f'ika-{n}'