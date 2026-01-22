import sys
from math import trunc
from typing import (
class SamiLocale(Locale):
    names = ['se', 'se-fi', 'se-no', 'se-se']
    past = '{0} dassái'
    future = '{0} '
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'dál', 'second': 'sekunda', 'seconds': '{0} sekundda', 'minute': 'minuhta', 'minutes': '{0} minuhta', 'hour': 'diimmu', 'hours': '{0} diimmu', 'day': 'beaivvi', 'days': '{0} beaivvi', 'week': 'vahku', 'weeks': '{0} vahku', 'month': 'mánu', 'months': '{0} mánu', 'year': 'jagi', 'years': '{0} jagi'}
    month_names = ['', 'Ođđajagimánnu', 'Guovvamánnu', 'Njukčamánnu', 'Cuoŋománnu', 'Miessemánnu', 'Geassemánnu', 'Suoidnemánnu', 'Borgemánnu', 'Čakčamánnu', 'Golggotmánnu', 'Skábmamánnu', 'Juovlamánnu']
    month_abbreviations = ['', 'Ođđajagimánnu', 'Guovvamánnu', 'Njukčamánnu', 'Cuoŋománnu', 'Miessemánnu', 'Geassemánnu', 'Suoidnemánnu', 'Borgemánnu', 'Čakčamánnu', 'Golggotmánnu', 'Skábmamánnu', 'Juovlamánnu']
    day_names = ['', 'Mánnodat', 'Disdat', 'Gaskavahkku', 'Duorastat', 'Bearjadat', 'Lávvordat', 'Sotnabeaivi']
    day_abbreviations = ['', 'Mánnodat', 'Disdat', 'Gaskavahkku', 'Duorastat', 'Bearjadat', 'Lávvordat', 'Sotnabeaivi']