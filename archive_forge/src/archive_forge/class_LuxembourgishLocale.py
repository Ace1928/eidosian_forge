import sys
from math import trunc
from typing import (
class LuxembourgishLocale(Locale):
    names = ['lb', 'lb-lu']
    past = 'virun {0}'
    future = 'an {0}'
    and_word = 'an'
    timeframes: ClassVar[Dict[TimeFrameLiteral, str]] = {'now': 'just elo', 'second': 'enger Sekonn', 'seconds': '{0} Sekonnen', 'minute': 'enger Minutt', 'minutes': '{0} Minutten', 'hour': 'enger Stonn', 'hours': '{0} Stonnen', 'day': 'engem Dag', 'days': '{0} Deeg', 'week': 'enger Woch', 'weeks': '{0} Wochen', 'month': 'engem Mount', 'months': '{0} Méint', 'year': 'engem Joer', 'years': '{0} Jahren'}
    timeframes_only_distance = timeframes.copy()
    timeframes_only_distance['second'] = 'eng Sekonn'
    timeframes_only_distance['minute'] = 'eng Minutt'
    timeframes_only_distance['hour'] = 'eng Stonn'
    timeframes_only_distance['day'] = 'een Dag'
    timeframes_only_distance['days'] = '{0} Deeg'
    timeframes_only_distance['week'] = 'eng Woch'
    timeframes_only_distance['month'] = 'ee Mount'
    timeframes_only_distance['months'] = '{0} Méint'
    timeframes_only_distance['year'] = 'ee Joer'
    timeframes_only_distance['years'] = '{0} Joer'
    month_names = ['', 'Januar', 'Februar', 'Mäerz', 'Abrëll', 'Mee', 'Juni', 'Juli', 'August', 'September', 'Oktouber', 'November', 'Dezember']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mäe', 'Abr', 'Mee', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    day_names = ['', 'Méindeg', 'Dënschdeg', 'Mëttwoch', 'Donneschdeg', 'Freideg', 'Samschdeg', 'Sonndeg']
    day_abbreviations = ['', 'Méi', 'Dën', 'Mët', 'Don', 'Fre', 'Sam', 'Son']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[int, float]=0, only_distance: bool=False) -> str:
        if not only_distance:
            return super().describe(timeframe, delta, only_distance)
        humanized: str = self.timeframes_only_distance[timeframe].format(trunc(abs(delta)))
        return humanized