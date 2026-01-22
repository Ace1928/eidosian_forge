import sys
from math import trunc
from typing import (
class AmharicLocale(Locale):
    names = ['am', 'am-et']
    past = '{0} በፊት'
    future = '{0} ውስጥ'
    and_word = 'እና'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[Mapping[str, str], str]]] = {'now': 'አሁን', 'second': {'past': 'ከአንድ ሰከንድ', 'future': 'በአንድ ሰከንድ'}, 'seconds': {'past': 'ከ {0} ሰከንድ', 'future': 'በ {0} ሰከንድ'}, 'minute': {'past': 'ከአንድ ደቂቃ', 'future': 'በአንድ ደቂቃ'}, 'minutes': {'past': 'ከ {0} ደቂቃዎች', 'future': 'በ {0} ደቂቃዎች'}, 'hour': {'past': 'ከአንድ ሰዓት', 'future': 'በአንድ ሰዓት'}, 'hours': {'past': 'ከ {0} ሰዓታት', 'future': 'በ {0} ሰከንድ'}, 'day': {'past': 'ከአንድ ቀን', 'future': 'በአንድ ቀን'}, 'days': {'past': 'ከ {0} ቀናት', 'future': 'በ {0} ቀናት'}, 'week': {'past': 'ከአንድ ሳምንት', 'future': 'በአንድ ሳምንት'}, 'weeks': {'past': 'ከ {0} ሳምንታት', 'future': 'በ {0} ሳምንታት'}, 'month': {'past': 'ከአንድ ወር', 'future': 'በአንድ ወር'}, 'months': {'past': 'ከ {0} ወር', 'future': 'በ {0} ወራት'}, 'year': {'past': 'ከአንድ አመት', 'future': 'በአንድ አመት'}, 'years': {'past': 'ከ {0} ዓመታት', 'future': 'በ {0} ዓመታት'}}
    timeframes_only_distance = {'second': 'አንድ ሰከንድ', 'seconds': '{0} ሰከንድ', 'minute': 'አንድ ደቂቃ', 'minutes': '{0} ደቂቃዎች', 'hour': 'አንድ ሰዓት', 'hours': '{0} ሰዓት', 'day': 'አንድ ቀን', 'days': '{0} ቀናት', 'week': 'አንድ ሳምንት', 'weeks': '{0} ሳምንት', 'month': 'አንድ ወር', 'months': '{0} ወራት', 'year': 'አንድ አመት', 'years': '{0} ዓመታት'}
    month_names = ['', 'ጃንዩወሪ', 'ፌብሩወሪ', 'ማርች', 'ኤፕሪል', 'ሜይ', 'ጁን', 'ጁላይ', 'ኦገስት', 'ሴፕቴምበር', 'ኦክቶበር', 'ኖቬምበር', 'ዲሴምበር']
    month_abbreviations = ['', 'ጃንዩ', 'ፌብሩ', 'ማርች', 'ኤፕሪ', 'ሜይ', 'ጁን', 'ጁላይ', 'ኦገስ', 'ሴፕቴ', 'ኦክቶ', 'ኖቬም', 'ዲሴም']
    day_names = ['', 'ሰኞ', 'ማክሰኞ', 'ረቡዕ', 'ሐሙስ', 'ዓርብ', 'ቅዳሜ', 'እሑድ']
    day_abbreviations = ['', 'እ', 'ሰ', 'ማ', 'ረ', 'ሐ', 'ዓ', 'ቅ']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}ኛ'

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        """
        Amharic awares time frame format function, takes into account
        the differences between general, past, and future forms (three different suffixes).
        """
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

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[float, int]=1, only_distance: bool=False) -> str:
        """Describes a delta within a timeframe in plain language.

        :param timeframe: a string representing a timeframe.
        :param delta: a quantity representing a delta in a timeframe.
        :param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords
        """
        if not only_distance:
            return super().describe(timeframe, delta, only_distance)
        humanized = self.timeframes_only_distance[timeframe].format(trunc(abs(delta)))
        return humanized