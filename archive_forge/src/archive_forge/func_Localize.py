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
def Localize(cls, obj):
    """If obj is naive, localize it to cls.LocalTimezone."""
    if not obj.tzinfo:
        return cls.LocalTimezone.localize(obj)
    return obj