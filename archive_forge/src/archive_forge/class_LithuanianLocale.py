import sys
from math import trunc
from typing import (
class LithuanianLocale(Locale):
    names = ['lt', 'lt-lt']
    past = 'prieš {0}'
    future = 'po {0}'
    and_word = 'ir'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'dabar', 'second': 'sekundės', 'seconds': '{0} sekundžių', 'minute': 'minutės', 'minutes': '{0} minučių', 'hour': 'valandos', 'hours': '{0} valandų', 'day': 'dieną', 'days': '{0} dienų', 'week': 'savaitės', 'weeks': '{0} savaičių', 'month': 'mėnesio', 'months': '{0} mėnesių', 'year': 'metų', 'years': '{0} metų'}
    month_names = ['', 'sausis', 'vasaris', 'kovas', 'balandis', 'gegužė', 'birželis', 'liepa', 'rugpjūtis', 'rugsėjis', 'spalis', 'lapkritis', 'gruodis']
    month_abbreviations = ['', 'saus', 'vas', 'kovas', 'bal', 'geg', 'birž', 'liepa', 'rugp', 'rugs', 'spalis', 'lapkr', 'gr']
    day_names = ['', 'pirmadienis', 'antradienis', 'trečiadienis', 'ketvirtadienis', 'penktadienis', 'šeštadienis', 'sekmadienis']
    day_abbreviations = ['', 'pi', 'an', 'tr', 'ke', 'pe', 'še', 'se']