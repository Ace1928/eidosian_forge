import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
class TimestampError(ValueError):
    """Generic timestamp-related error."""
    pass