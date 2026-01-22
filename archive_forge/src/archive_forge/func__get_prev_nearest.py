from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
def _get_prev_nearest(self, x, to_check):
    small = [item for item in to_check if item <= x]
    large = [item for item in to_check if item > x]
    small.reverse()
    large.reverse()
    small.extend(large)
    return small[0]