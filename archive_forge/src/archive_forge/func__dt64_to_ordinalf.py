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
def _dt64_to_ordinalf(d):
    """
    Convert `numpy.datetime64` or an `numpy.ndarray` of those types to
    Gregorian date as UTC float relative to the epoch (see `.get_epoch`).
    Roundoff is float64 precision.  Practically: microseconds for dates
    between 290301 BC, 294241 AD, milliseconds for larger dates
    (see `numpy.datetime64`).
    """
    dseconds = d.astype('datetime64[s]')
    extra = (d - dseconds).astype('timedelta64[ns]')
    t0 = np.datetime64(get_epoch(), 's')
    dt = (dseconds - t0).astype(np.float64)
    dt += extra.astype(np.float64) / 1000000000.0
    dt = dt / SEC_PER_DAY
    NaT_int = np.datetime64('NaT').astype(np.int64)
    d_int = d.astype(np.int64)
    dt[d_int == NaT_int] = np.nan
    return dt