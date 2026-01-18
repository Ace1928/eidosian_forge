from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_tz_convert_nat(self):
    dates = [NaT]
    idx = DatetimeIndex(dates)
    idx = idx.tz_localize('US/Pacific')
    tm.assert_index_equal(idx, DatetimeIndex(dates, tz='US/Pacific'))
    idx = idx.tz_convert('US/Eastern')
    tm.assert_index_equal(idx, DatetimeIndex(dates, tz='US/Eastern'))
    idx = idx.tz_convert('UTC')
    tm.assert_index_equal(idx, DatetimeIndex(dates, tz='UTC'))
    dates = ['2010-12-01 00:00', '2010-12-02 00:00', NaT]
    idx = DatetimeIndex(dates)
    idx = idx.tz_localize('US/Pacific')
    tm.assert_index_equal(idx, DatetimeIndex(dates, tz='US/Pacific'))
    idx = idx.tz_convert('US/Eastern')
    expected = ['2010-12-01 03:00', '2010-12-02 03:00', NaT]
    tm.assert_index_equal(idx, DatetimeIndex(expected, tz='US/Eastern'))
    idx = idx + offsets.Hour(5)
    expected = ['2010-12-01 08:00', '2010-12-02 08:00', NaT]
    tm.assert_index_equal(idx, DatetimeIndex(expected, tz='US/Eastern'))
    idx = idx.tz_convert('US/Pacific')
    expected = ['2010-12-01 05:00', '2010-12-02 05:00', NaT]
    tm.assert_index_equal(idx, DatetimeIndex(expected, tz='US/Pacific'))
    idx = idx + np.timedelta64(3, 'h')
    expected = ['2010-12-01 08:00', '2010-12-02 08:00', NaT]
    tm.assert_index_equal(idx, DatetimeIndex(expected, tz='US/Pacific'))
    idx = idx.tz_convert('US/Eastern')
    expected = ['2010-12-01 11:00', '2010-12-02 11:00', NaT]
    tm.assert_index_equal(idx, DatetimeIndex(expected, tz='US/Eastern'))