import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
class TimezoneNotSpecifiedError(TimestampError):
    """This error is raised when timezone is not specified."""
    pass