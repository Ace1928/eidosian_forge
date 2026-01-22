import sys
from math import trunc
from typing import (
class ZuluLocale(Locale):
    names = ['zu', 'zu-za']
    past = '{0} edlule'
    future = '{0} '
    and_word = 'futhi'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[Mapping[str, str], str]]] = {'now': 'manje', 'second': {'past': 'umzuzwana', 'future': 'ngomzuzwana'}, 'seconds': {'past': '{0} imizuzwana', 'future': '{0} ngemizuzwana'}, 'minute': {'past': 'umzuzu', 'future': 'ngomzuzu'}, 'minutes': {'past': '{0} imizuzu', 'future': '{0} ngemizuzu'}, 'hour': {'past': 'ihora', 'future': 'ngehora'}, 'hours': {'past': '{0} amahora', 'future': '{0} emahoreni'}, 'day': {'past': 'usuku', 'future': 'ngosuku'}, 'days': {'past': '{0} izinsuku', 'future': '{0} ezinsukwini'}, 'week': {'past': 'isonto', 'future': 'ngesonto'}, 'weeks': {'past': '{0} amasonto', 'future': '{0} emasontweni'}, 'month': {'past': 'inyanga', 'future': 'ngenyanga'}, 'months': {'past': '{0} izinyanga', 'future': '{0} ezinyangeni'}, 'year': {'past': 'unyaka', 'future': 'ngonyak'}, 'years': {'past': '{0} iminyaka', 'future': '{0} eminyakeni'}}

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        """Zulu aware time frame format function, takes into account
        the differences between past and future forms."""
        abs_delta = abs(delta)
        form = self.timeframes[timeframe]
        if isinstance(form, str):
            return form.format(abs_delta)
        if delta > 0:
            key = 'future'
        else:
            key = 'past'
        form = form[key]
        return form.format(abs_delta)
    month_names = ['', 'uMasingane', 'uNhlolanja', 'uNdasa', 'UMbasa', 'UNhlaba', 'UNhlangulana', 'uNtulikazi', 'UNcwaba', 'uMandulo', 'uMfumfu', 'uLwezi', 'uZibandlela']
    month_abbreviations = ['', 'uMasingane', 'uNhlolanja', 'uNdasa', 'UMbasa', 'UNhlaba', 'UNhlangulana', 'uNtulikazi', 'UNcwaba', 'uMandulo', 'uMfumfu', 'uLwezi', 'uZibandlela']
    day_names = ['', 'uMsombuluko', 'uLwesibili', 'uLwesithathu', 'uLwesine', 'uLwesihlanu', 'uMgqibelo', 'iSonto']
    day_abbreviations = ['', 'uMsombuluko', 'uLwesibili', 'uLwesithathu', 'uLwesine', 'uLwesihlanu', 'uMgqibelo', 'iSonto']