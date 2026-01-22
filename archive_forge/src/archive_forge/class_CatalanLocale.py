import sys
from math import trunc
from typing import (
class CatalanLocale(Locale):
    names = ['ca', 'ca-es', 'ca-ad', 'ca-fr', 'ca-it']
    past = 'Fa {0}'
    future = 'En {0}'
    and_word = 'i'
    timeframes = {'now': 'Ara mateix', 'second': 'un segon', 'seconds': '{0} segons', 'minute': 'un minut', 'minutes': '{0} minuts', 'hour': 'una hora', 'hours': '{0} hores', 'day': 'un dia', 'days': '{0} dies', 'month': 'un mes', 'months': '{0} mesos', 'year': 'un any', 'years': '{0} anys'}
    month_names = ['', 'gener', 'febrer', 'març', 'abril', 'maig', 'juny', 'juliol', 'agost', 'setembre', 'octubre', 'novembre', 'desembre']
    month_abbreviations = ['', 'gen.', 'febr.', 'març', 'abr.', 'maig', 'juny', 'jul.', 'ag.', 'set.', 'oct.', 'nov.', 'des.']
    day_names = ['', 'dilluns', 'dimarts', 'dimecres', 'dijous', 'divendres', 'dissabte', 'diumenge']
    day_abbreviations = ['', 'dl.', 'dt.', 'dc.', 'dj.', 'dv.', 'ds.', 'dg.']