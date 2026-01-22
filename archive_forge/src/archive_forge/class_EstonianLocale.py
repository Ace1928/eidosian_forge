import sys
from math import trunc
from typing import (
class EstonianLocale(Locale):
    names = ['ee', 'et']
    past = '{0} tagasi'
    future = '{0} pärast'
    and_word = 'ja'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Mapping[str, str]]] = {'now': {'past': 'just nüüd', 'future': 'just nüüd'}, 'second': {'past': 'üks sekund', 'future': 'ühe sekundi'}, 'seconds': {'past': '{0} sekundit', 'future': '{0} sekundi'}, 'minute': {'past': 'üks minut', 'future': 'ühe minuti'}, 'minutes': {'past': '{0} minutit', 'future': '{0} minuti'}, 'hour': {'past': 'tund aega', 'future': 'tunni aja'}, 'hours': {'past': '{0} tundi', 'future': '{0} tunni'}, 'day': {'past': 'üks päev', 'future': 'ühe päeva'}, 'days': {'past': '{0} päeva', 'future': '{0} päeva'}, 'month': {'past': 'üks kuu', 'future': 'ühe kuu'}, 'months': {'past': '{0} kuud', 'future': '{0} kuu'}, 'year': {'past': 'üks aasta', 'future': 'ühe aasta'}, 'years': {'past': '{0} aastat', 'future': '{0} aasta'}}
    month_names = ['', 'Jaanuar', 'Veebruar', 'Märts', 'Aprill', 'Mai', 'Juuni', 'Juuli', 'August', 'September', 'Oktoober', 'November', 'Detsember']
    month_abbreviations = ['', 'Jan', 'Veb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dets']
    day_names = ['', 'Esmaspäev', 'Teisipäev', 'Kolmapäev', 'Neljapäev', 'Reede', 'Laupäev', 'Pühapäev']
    day_abbreviations = ['', 'Esm', 'Teis', 'Kolm', 'Nelj', 'Re', 'Lau', 'Püh']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        if delta > 0:
            _form = form['future']
        else:
            _form = form['past']
        return _form.format(abs(delta))