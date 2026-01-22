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
def _parse_offset(self, s):
    s = s.strip()
    if not s:
        raise ValueError('empty offset')
    if s[0] in ('+', '-'):
        signal = (-1, +1)[s[0] == '+']
        s = s[1:]
    else:
        signal = +1
    if len(s) == 4:
        return (int(s[:2]) * 3600 + int(s[2:]) * 60) * signal
    elif len(s) == 6:
        return (int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])) * signal
    else:
        raise ValueError('invalid offset: ' + s)