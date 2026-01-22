import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@staticmethod
def _date_generator(startdate, timedelta, iterations):
    currentdate = startdate
    currentiteration = 0
    while currentiteration < iterations:
        yield currentdate
        currentdate += timedelta
        currentiteration += 1