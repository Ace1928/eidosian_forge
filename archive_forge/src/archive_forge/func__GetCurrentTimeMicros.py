import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def _GetCurrentTimeMicros():
    """Get the current time in microseconds, in UTC.

  Returns:
    The number of microseconds since the epoch.
  """
    return int(SecondsToMicroseconds(time.time()))