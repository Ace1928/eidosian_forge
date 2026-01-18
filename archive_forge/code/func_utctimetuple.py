import time as _time
import math as _math
import sys
from operator import index as _index
def utctimetuple(self):
    """Return UTC time tuple compatible with time.gmtime()."""
    offset = self.utcoffset()
    if offset:
        self -= offset
    y, m, d = (self.year, self.month, self.day)
    hh, mm, ss = (self.hour, self.minute, self.second)
    return _build_struct_time(y, m, d, hh, mm, ss, 0)