import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
def wdayset(self, year, month, day):
    dset = [None] * (self.yearlen + 7)
    i = datetime.date(year, month, day).toordinal() - self.yearordinal
    start = i
    for j in range(7):
        dset[i] = i
        i += 1
        if self.wdaymask[i] == self.rrule._wkst:
            break
    return (dset, start, i)