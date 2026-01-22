import sys
from math import trunc
from typing import (
class LatvianLocale(Locale):
    names = ['lv', 'lv-lv']
    past = 'pirms {0}'
    future = 'pēc {0}'
    and_word = 'un'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'tagad', 'second': 'sekundes', 'seconds': '{0} sekundēm', 'minute': 'minūtes', 'minutes': '{0} minūtēm', 'hour': 'stundas', 'hours': '{0} stundām', 'day': 'dienas', 'days': '{0} dienām', 'week': 'nedēļas', 'weeks': '{0} nedēļām', 'month': 'mēneša', 'months': '{0} mēnešiem', 'year': 'gada', 'years': '{0} gadiem'}
    month_names = ['', 'janvāris', 'februāris', 'marts', 'aprīlis', 'maijs', 'jūnijs', 'jūlijs', 'augusts', 'septembris', 'oktobris', 'novembris', 'decembris']
    month_abbreviations = ['', 'jan', 'feb', 'marts', 'apr', 'maijs', 'jūnijs', 'jūlijs', 'aug', 'sept', 'okt', 'nov', 'dec']
    day_names = ['', 'pirmdiena', 'otrdiena', 'trešdiena', 'ceturtdiena', 'piektdiena', 'sestdiena', 'svētdiena']
    day_abbreviations = ['', 'pi', 'ot', 'tr', 'ce', 'pi', 'se', 'sv']