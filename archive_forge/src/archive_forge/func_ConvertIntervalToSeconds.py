import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def ConvertIntervalToSeconds(interval):
    """Convert a formatted string representing an interval into seconds.

  Args:
    interval: String to interpret as an interval.  A basic interval looks like
      "<number><suffix>".  Complex intervals consisting of a chain of basic
      intervals are also allowed.

  Returns:
    An integer representing the number of seconds represented by the interval
    string, or None if the interval string could not be decoded.
  """
    total = 0
    while interval:
        match = _INTERVAL_REGEXP.match(interval)
        if not match:
            return None
        try:
            num = int(match.group(1))
        except ValueError:
            return None
        suffix = match.group(2)
        if suffix:
            multiplier = _INTERVAL_CONV_DICT.get(suffix)
            if not multiplier:
                return None
            num *= multiplier
        total += num
        interval = interval[match.end(0):]
    return total