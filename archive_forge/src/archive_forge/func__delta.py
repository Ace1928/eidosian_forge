from relative deltas), local machine timezone, fixed offset timezone, and UTC
import datetime
import logging  # GOOGLE
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict
import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs
from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
from warnings import warn
def _delta(self, x, isend=0):
    from dateutil import relativedelta
    kwargs = {}
    if x.month is not None:
        kwargs['month'] = x.month
        if x.weekday is not None:
            kwargs['weekday'] = relativedelta.weekday(x.weekday, x.week)
            if x.week > 0:
                kwargs['day'] = 1
            else:
                kwargs['day'] = 31
        elif x.day:
            kwargs['day'] = x.day
    elif x.yday is not None:
        kwargs['yearday'] = x.yday
    elif x.jyday is not None:
        kwargs['nlyearday'] = x.jyday
    if not kwargs:
        if not isend:
            kwargs['month'] = 4
            kwargs['day'] = 1
            kwargs['weekday'] = relativedelta.SU(+1)
        else:
            kwargs['month'] = 10
            kwargs['day'] = 31
            kwargs['weekday'] = relativedelta.SU(-1)
    if x.time is not None:
        kwargs['seconds'] = x.time
    else:
        kwargs['seconds'] = 7200
    if isend:
        delta = self._dst_offset - self._std_offset
        kwargs['seconds'] -= delta.seconds + delta.days * 86400
    return relativedelta.relativedelta(**kwargs)