import sys
from math import trunc
from typing import (
class AlbanianLocale(Locale):
    names = ['sq', 'sq-al']
    past = '{0} më parë'
    future = 'në {0}'
    and_word = 'dhe'
    timeframes = {'now': 'tani', 'second': 'sekondë', 'seconds': '{0} sekonda', 'minute': 'minutë', 'minutes': '{0} minuta', 'hour': 'orë', 'hours': '{0} orë', 'day': 'ditë', 'days': '{0} ditë', 'week': 'javë', 'weeks': '{0} javë', 'month': 'muaj', 'months': '{0} muaj', 'year': 'vit', 'years': '{0} vjet'}
    month_names = ['', 'janar', 'shkurt', 'mars', 'prill', 'maj', 'qershor', 'korrik', 'gusht', 'shtator', 'tetor', 'nëntor', 'dhjetor']
    month_abbreviations = ['', 'jan', 'shk', 'mar', 'pri', 'maj', 'qer', 'korr', 'gush', 'sht', 'tet', 'nën', 'dhj']
    day_names = ['', 'e hënë', 'e martë', 'e mërkurë', 'e enjte', 'e premte', 'e shtunë', 'e diel']
    day_abbreviations = ['', 'hën', 'mar', 'mër', 'enj', 'pre', 'sht', 'die']