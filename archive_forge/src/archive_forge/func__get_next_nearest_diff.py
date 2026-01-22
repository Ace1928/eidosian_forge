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
def _get_next_nearest_diff(self, x, to_check, range_val):
    for i, d in enumerate(to_check):
        if d == 'l':
            d = range_val
        if d >= x:
            return d - x
    return to_check[0] - x + range_val