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
def utcfromtimestamp(cls, *args, **kwargs):
    """timestamp -> UTC datetime from a POSIX timestamp (like time.time())."""
    return super(BaseTimestamp, cls).utcfromtimestamp(*args, **kwargs).replace(tzinfo=pytz.utc)