import sys
from math import trunc
from typing import (
class LatinLocale(Locale):
    names = ['la', 'la-va']
    past = 'ante {0}'
    future = 'in {0}'
    and_word = 'et'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'nunc', 'second': 'secundum', 'seconds': '{0} secundis', 'minute': 'minutam', 'minutes': '{0} minutis', 'hour': 'horam', 'hours': '{0} horas', 'day': 'diem', 'days': '{0} dies', 'week': 'hebdomadem', 'weeks': '{0} hebdomades', 'month': 'mensem', 'months': '{0} mensis', 'year': 'annum', 'years': '{0} annos'}
    month_names = ['', 'Ianuarius', 'Februarius', 'Martius', 'Aprilis', 'Maius', 'Iunius', 'Iulius', 'Augustus', 'September', 'October', 'November', 'December']
    month_abbreviations = ['', 'Ian', 'Febr', 'Mart', 'Apr', 'Mai', 'Iun', 'Iul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    day_names = ['', 'dies Lunae', 'dies Martis', 'dies Mercurii', 'dies Iovis', 'dies Veneris', 'dies Saturni', 'dies Solis']
    day_abbreviations = ['', 'dies Lunae', 'dies Martis', 'dies Mercurii', 'dies Iovis', 'dies Veneris', 'dies Saturni', 'dies Solis']