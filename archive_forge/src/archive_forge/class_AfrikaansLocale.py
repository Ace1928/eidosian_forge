import sys
from math import trunc
from typing import (
class AfrikaansLocale(Locale):
    names = ['af', 'af-nl']
    past = '{0} gelede'
    future = 'in {0}'
    timeframes = {'now': 'nou', 'second': 'n sekonde', 'seconds': '{0} sekondes', 'minute': 'minuut', 'minutes': '{0} minute', 'hour': 'uur', 'hours': '{0} ure', 'day': 'een dag', 'days': '{0} dae', 'month': 'een maand', 'months': '{0} maande', 'year': 'een jaar', 'years': '{0} jaar'}
    month_names = ['', 'Januarie', 'Februarie', 'Maart', 'April', 'Mei', 'Junie', 'Julie', 'Augustus', 'September', 'Oktober', 'November', 'Desember']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Des']
    day_names = ['', 'Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrydag', 'Saterdag', 'Sondag']
    day_abbreviations = ['', 'Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'So']