import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
def test_properties_business(self):
    freq = to_offset('B')
    ts = Timestamp('2017-10-01')
    assert ts.dayofweek == 6
    assert ts.day_of_week == 6
    assert ts.is_month_start
    assert not freq.is_month_start(ts)
    assert freq.is_month_start(ts + Timedelta(days=1))
    assert not freq.is_quarter_start(ts)
    assert freq.is_quarter_start(ts + Timedelta(days=1))
    ts = Timestamp('2017-09-30')
    assert ts.dayofweek == 5
    assert ts.day_of_week == 5
    assert ts.is_month_end
    assert not freq.is_month_end(ts)
    assert freq.is_month_end(ts - Timedelta(days=1))
    assert ts.is_quarter_end
    assert not freq.is_quarter_end(ts)
    assert freq.is_quarter_end(ts - Timedelta(days=1))