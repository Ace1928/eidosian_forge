import time as _time
import math as _math
import sys
from operator import index as _index
def _tzstr(self):
    """Return formatted timezone offset (+xx:xx) or an empty string."""
    off = self.utcoffset()
    return _format_offset(off)