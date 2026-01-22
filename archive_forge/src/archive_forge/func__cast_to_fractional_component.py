import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def _cast_to_fractional_component(conversion, floatstr):
    intpart, floatpart = floatstr.split('.')
    intvalue = int(intpart)
    preconvertedvalue = int(floatpart)
    convertedvalue = preconvertedvalue * conversion // 10 ** len(floatpart)
    return FractionalComponent(intvalue, convertedvalue)