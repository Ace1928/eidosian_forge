import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
@classmethod
def today(cls):
    """Current BaseTimestamp.

    Same as self.__class__.fromtimestamp(time.time()).
    Returns:
      New self.__class__.
    """
    return cls.AddLocalTimezone(super(BaseTimestamp, cls).today())