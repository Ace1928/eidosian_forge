import sys
from math import trunc
from typing import (
class RomanshLocale(Locale):
    names = ['rm', 'rm-ch']
    past = 'avant {0}'
    future = 'en {0}'
    timeframes = {'now': 'en quest mument', 'second': 'in secunda', 'seconds': '{0} secundas', 'minute': 'ina minuta', 'minutes': '{0} minutas', 'hour': "in'ura", 'hours': '{0} ura', 'day': 'in di', 'days': '{0} dis', 'week': "in'emna", 'weeks': '{0} emnas', 'month': 'in mais', 'months': '{0} mais', 'year': 'in onn', 'years': '{0} onns'}
    month_names = ['', 'schaner', 'favrer', 'mars', 'avrigl', 'matg', 'zercladur', 'fanadur', 'avust', 'settember', 'october', 'november', 'december']
    month_abbreviations = ['', 'schan', 'fav', 'mars', 'avr', 'matg', 'zer', 'fan', 'avu', 'set', 'oct', 'nov', 'dec']
    day_names = ['', 'glindesdi', 'mardi', 'mesemna', 'gievgia', 'venderdi', 'sonda', 'dumengia']
    day_abbreviations = ['', 'gli', 'ma', 'me', 'gie', 've', 'so', 'du']