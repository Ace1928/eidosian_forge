import sys
from math import trunc
from typing import (
class HungarianLocale(Locale):
    names = ['hu', 'hu-hu']
    past = '{0} ezelőtt'
    future = '{0} múlva'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'éppen most', 'second': {'past': 'egy második', 'future': 'egy második'}, 'seconds': {'past': '{0} másodpercekkel', 'future': '{0} pár másodperc'}, 'minute': {'past': 'egy perccel', 'future': 'egy perc'}, 'minutes': {'past': '{0} perccel', 'future': '{0} perc'}, 'hour': {'past': 'egy órával', 'future': 'egy óra'}, 'hours': {'past': '{0} órával', 'future': '{0} óra'}, 'day': {'past': 'egy nappal', 'future': 'egy nap'}, 'days': {'past': '{0} nappal', 'future': '{0} nap'}, 'week': {'past': 'egy héttel', 'future': 'egy hét'}, 'weeks': {'past': '{0} héttel', 'future': '{0} hét'}, 'month': {'past': 'egy hónappal', 'future': 'egy hónap'}, 'months': {'past': '{0} hónappal', 'future': '{0} hónap'}, 'year': {'past': 'egy évvel', 'future': 'egy év'}, 'years': {'past': '{0} évvel', 'future': '{0} év'}}
    month_names = ['', 'január', 'február', 'március', 'április', 'május', 'június', 'július', 'augusztus', 'szeptember', 'október', 'november', 'december']
    month_abbreviations = ['', 'jan', 'febr', 'márc', 'ápr', 'máj', 'jún', 'júl', 'aug', 'szept', 'okt', 'nov', 'dec']
    day_names = ['', 'hétfő', 'kedd', 'szerda', 'csütörtök', 'péntek', 'szombat', 'vasárnap']
    day_abbreviations = ['', 'hét', 'kedd', 'szer', 'csüt', 'pént', 'szom', 'vas']
    meridians = {'am': 'de', 'pm': 'du', 'AM': 'DE', 'PM': 'DU'}

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        if isinstance(form, Mapping):
            if delta > 0:
                form = form['future']
            else:
                form = form['past']
        return form.format(abs(delta))