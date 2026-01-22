import sys
from math import trunc
from typing import (
class DutchLocale(Locale):
    names = ['nl', 'nl-nl']
    past = '{0} geleden'
    future = 'over {0}'
    timeframes = {'now': 'nu', 'second': 'een seconde', 'seconds': '{0} seconden', 'minute': 'een minuut', 'minutes': '{0} minuten', 'hour': 'een uur', 'hours': '{0} uur', 'day': 'een dag', 'days': '{0} dagen', 'week': 'een week', 'weeks': '{0} weken', 'month': 'een maand', 'months': '{0} maanden', 'year': 'een jaar', 'years': '{0} jaar'}
    month_names = ['', 'januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december']
    month_abbreviations = ['', 'jan', 'feb', 'mrt', 'apr', 'mei', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
    day_abbreviations = ['', 'ma', 'di', 'wo', 'do', 'vr', 'za', 'zo']