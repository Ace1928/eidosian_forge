from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def _build_tzinfo(self, tzinfos, tzname, tzoffset):
    if callable(tzinfos):
        tzdata = tzinfos(tzname, tzoffset)
    else:
        tzdata = tzinfos.get(tzname)
    if isinstance(tzdata, datetime.tzinfo) or tzdata is None:
        tzinfo = tzdata
    elif isinstance(tzdata, text_type):
        tzinfo = tz.tzstr(tzdata)
    elif isinstance(tzdata, integer_types):
        tzinfo = tz.tzoffset(tzname, tzdata)
    else:
        raise TypeError('Offset must be tzinfo subclass, tz string, or int offset.')
    return tzinfo