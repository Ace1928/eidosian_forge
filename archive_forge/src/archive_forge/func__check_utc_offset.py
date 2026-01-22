import time as _time
import math as _math
import sys
from operator import index as _index
def _check_utc_offset(name, offset):
    assert name in ('utcoffset', 'dst')
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.%s() must return None or timedelta, not '%s'" % (name, type(offset)))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError('%s()=%s, must be strictly between -timedelta(hours=24) and timedelta(hours=24)' % (name, offset))