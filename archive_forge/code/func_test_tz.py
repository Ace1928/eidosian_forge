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
def test_tz(self):
    tstr = '2014-02-01 09:00'
    ts = Timestamp(tstr)
    local = ts.tz_localize('Asia/Tokyo')
    assert local.hour == 9
    assert local == Timestamp(tstr, tz='Asia/Tokyo')
    conv = local.tz_convert('US/Eastern')
    assert conv == Timestamp('2014-01-31 19:00', tz='US/Eastern')
    assert conv.hour == 19
    ts = Timestamp(tstr) + offsets.Nano(5)
    local = ts.tz_localize('Asia/Tokyo')
    assert local.hour == 9
    assert local.nanosecond == 5
    conv = local.tz_convert('US/Eastern')
    assert conv.nanosecond == 5
    assert conv.hour == 19