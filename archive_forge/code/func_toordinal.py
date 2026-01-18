import time as _time
import math as _math
import sys
from operator import index as _index
def toordinal(self):
    """Return proleptic Gregorian ordinal for the year, month and day.

        January 1 of year 1 is day 1.  Only the year, month and day values
        contribute to the result.
        """
    return _ymd2ord(self._year, self._month, self._day)