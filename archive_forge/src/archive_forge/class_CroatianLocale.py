import sys
from math import trunc
from typing import (
class CroatianLocale(Locale):
    names = ['hr', 'hr-hr']
    past = 'prije {0}'
    future = 'za {0}'
    and_word = 'i'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'upravo sad', 'second': 'sekundu', 'seconds': {'double': '{0} sekunde', 'higher': '{0} sekundi'}, 'minute': 'minutu', 'minutes': {'double': '{0} minute', 'higher': '{0} minuta'}, 'hour': 'sat', 'hours': {'double': '{0} sata', 'higher': '{0} sati'}, 'day': 'jedan dan', 'days': {'double': '{0} dana', 'higher': '{0} dana'}, 'week': 'tjedan', 'weeks': {'double': '{0} tjedna', 'higher': '{0} tjedana'}, 'month': 'mjesec', 'months': {'double': '{0} mjeseca', 'higher': '{0} mjeseci'}, 'year': 'godinu', 'years': {'double': '{0} godine', 'higher': '{0} godina'}}
    month_names = ['', 'siječanj', 'veljača', 'ožujak', 'travanj', 'svibanj', 'lipanj', 'srpanj', 'kolovoz', 'rujan', 'listopad', 'studeni', 'prosinac']
    month_abbreviations = ['', 'siječ', 'velj', 'ožuj', 'trav', 'svib', 'lip', 'srp', 'kol', 'ruj', 'list', 'stud', 'pros']
    day_names = ['', 'ponedjeljak', 'utorak', 'srijeda', 'četvrtak', 'petak', 'subota', 'nedjelja']
    day_abbreviations = ['', 'po', 'ut', 'sr', 'če', 'pe', 'su', 'ne']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if 1 < delta <= 4:
                form = form['double']
            else:
                form = form['higher']
        return form.format(delta)