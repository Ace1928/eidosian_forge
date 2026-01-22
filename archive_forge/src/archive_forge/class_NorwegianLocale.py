import sys
from math import trunc
from typing import (
class NorwegianLocale(Locale):
    names = ['nb', 'nb-no']
    past = 'for {0} siden'
    future = 'om {0}'
    timeframes = {'now': 'nå nettopp', 'second': 'ett sekund', 'seconds': '{0} sekunder', 'minute': 'ett minutt', 'minutes': '{0} minutter', 'hour': 'en time', 'hours': '{0} timer', 'day': 'en dag', 'days': '{0} dager', 'week': 'en uke', 'weeks': '{0} uker', 'month': 'en måned', 'months': '{0} måneder', 'year': 'ett år', 'years': '{0} år'}
    month_names = ['', 'januar', 'februar', 'mars', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'oktober', 'november', 'desember']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']
    day_names = ['', 'mandag', 'tirsdag', 'onsdag', 'torsdag', 'fredag', 'lørdag', 'søndag']
    day_abbreviations = ['', 'ma', 'ti', 'on', 'to', 'fr', 'lø', 'sø']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'