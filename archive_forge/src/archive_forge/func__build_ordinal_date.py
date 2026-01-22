import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _build_ordinal_date(isoyear, isoday):
    builtdate = datetime.date(isoyear, 1, 1) + datetime.timedelta(days=isoday - 1)
    return builtdate