from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('scalar', [False, pytest.param(True, marks=pytest.mark.xfail(reason='GH#56410 scalar case not yet addressed'))])
def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar):
    tdi = date_range('2016-01-01', periods=3, unit='s') - Timestamp('1970-01-01')
    item = Timestamp('2016-02-03 04:05:06.789') - Timestamp('1970-01-01')
    vec = timedelta_range(item, periods=3, unit='ms')
    expected = Series([item, tdi[1], tdi[2]], dtype='m8[ms]')
    ser = Series(tdi)
    ser[0] = NaT
    ser2 = ser.copy()
    res = ser.fillna(item)
    res2 = ser2.fillna(Series(vec))
    if scalar:
        tm.assert_series_equal(res, expected)
    else:
        tm.assert_series_equal(res2, expected)