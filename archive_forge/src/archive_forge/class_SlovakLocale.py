import sys
from math import trunc
from typing import (
class SlovakLocale(Locale):
    names = ['sk', 'sk-sk']
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'Teraz', 'second': {'past': 'sekundou', 'future': 'sekundu'}, 'seconds': {'zero': '{0} sekúnd', 'past': '{0} sekundami', 'future-singular': '{0} sekundy', 'future-paucal': '{0} sekúnd'}, 'minute': {'past': 'minútou', 'future': 'minútu'}, 'minutes': {'zero': '{0} minút', 'past': '{0} minútami', 'future-singular': '{0} minúty', 'future-paucal': '{0} minút'}, 'hour': {'past': 'hodinou', 'future': 'hodinu'}, 'hours': {'zero': '{0} hodín', 'past': '{0} hodinami', 'future-singular': '{0} hodiny', 'future-paucal': '{0} hodín'}, 'day': {'past': 'dňom', 'future': 'deň'}, 'days': {'zero': '{0} dní', 'past': '{0} dňami', 'future-singular': '{0} dni', 'future-paucal': '{0} dní'}, 'week': {'past': 'týždňom', 'future': 'týždeň'}, 'weeks': {'zero': '{0} týždňov', 'past': '{0} týždňami', 'future-singular': '{0} týždne', 'future-paucal': '{0} týždňov'}, 'month': {'past': 'mesiacom', 'future': 'mesiac'}, 'months': {'zero': '{0} mesiacov', 'past': '{0} mesiacmi', 'future-singular': '{0} mesiace', 'future-paucal': '{0} mesiacov'}, 'year': {'past': 'rokom', 'future': 'rok'}, 'years': {'zero': '{0} rokov', 'past': '{0} rokmi', 'future-singular': '{0} roky', 'future-paucal': '{0} rokov'}}
    past = 'Pred {0}'
    future = 'O {0}'
    and_word = 'a'
    month_names = ['', 'január', 'február', 'marec', 'apríl', 'máj', 'jún', 'júl', 'august', 'september', 'október', 'november', 'december']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'máj', 'jún', 'júl', 'aug', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'pondelok', 'utorok', 'streda', 'štvrtok', 'piatok', 'sobota', 'nedeľa']
    day_abbreviations = ['', 'po', 'ut', 'st', 'št', 'pi', 'so', 'ne']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        """Slovak aware time frame format function, takes into account
        the differences between past and future forms."""
        abs_delta = abs(delta)
        form = self.timeframes[timeframe]
        if isinstance(form, str):
            return form.format(abs_delta)
        if delta == 0:
            key = 'zero'
        elif delta < 0:
            key = 'past'
        elif 'future-singular' not in form:
            key = 'future'
        elif 2 <= abs_delta % 10 <= 4 and (abs_delta % 100 < 10 or abs_delta % 100 >= 20):
            key = 'future-singular'
        else:
            key = 'future-paucal'
        form: str = form[key]
        return form.format(abs_delta)