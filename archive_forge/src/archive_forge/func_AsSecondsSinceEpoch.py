import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
def AsSecondsSinceEpoch(self):
    """Return number of seconds since epoch (timestamp in seconds)."""
    return GetSecondsSinceEpoch(self.utctimetuple())