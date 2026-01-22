import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def _attach_tzinfo(self, dt, tzinfo):
    if hasattr(tzinfo, 'localize'):
        return tzinfo.localize(dt, is_dst=True)
    return dt.replace(tzinfo=tzinfo)