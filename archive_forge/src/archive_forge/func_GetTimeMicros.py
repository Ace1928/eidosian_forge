import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def GetTimeMicros(time_tuple):
    """Get a time in microseconds.

  Arguments:
    time_tuple: A (year, month, day, hour, minute, second) tuple (the python
      time tuple format) in the UTC time zone.

  Returns:
    The number of microseconds since the epoch represented by the input tuple.
  """
    return int(SecondsToMicroseconds(GetSecondsSinceEpoch(time_tuple)))