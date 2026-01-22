import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _date_generator_unbounded(startdate, timedelta):
    currentdate = startdate
    while True:
        yield currentdate
        currentdate += timedelta