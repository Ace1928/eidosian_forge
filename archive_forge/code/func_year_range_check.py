import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def year_range_check(valuestr, limit):
    YYYYstr = valuestr
    if len(valuestr) < 4:
        YYYYstr = valuestr.ljust(4, '0')
    return range_check(YYYYstr, limit)