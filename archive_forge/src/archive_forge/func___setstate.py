import time as _time
import math as _math
import sys
from operator import index as _index
def __setstate(self, string, tzinfo):
    if tzinfo is not None and (not isinstance(tzinfo, _tzinfo_class)):
        raise TypeError('bad tzinfo state arg')
    yhi, ylo, m, self._day, self._hour, self._minute, self._second, us1, us2, us3 = string
    if m > 127:
        self._fold = 1
        self._month = m - 128
    else:
        self._fold = 0
        self._month = m
    self._year = yhi * 256 + ylo
    self._microsecond = (us1 << 8 | us2) << 8 | us3
    self._tzinfo = tzinfo