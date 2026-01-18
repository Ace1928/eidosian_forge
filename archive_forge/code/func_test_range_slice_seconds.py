import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('make_range', [date_range, period_range])
def test_range_slice_seconds(self, make_range):
    idx = make_range(start='2013/01/01 09:00:00', freq='s', periods=4000)
    msg = 'slice indices must be integers or None or have an __index__ method'
    values = ['2014', '2013/02', '2013/01/02', '2013/02/01 9H', '2013/02/01 09:00']
    for v in values:
        with pytest.raises(TypeError, match=msg):
            idx[v:]
    s = Series(np.random.default_rng(2).random(len(idx)), index=idx)
    tm.assert_series_equal(s['2013/01/01 09:05':'2013/01/01 09:10'], s[300:660])
    tm.assert_series_equal(s['2013/01/01 10:00':'2013/01/01 10:05'], s[3600:3960])
    tm.assert_series_equal(s['2013/01/01 10H':], s[3600:])
    tm.assert_series_equal(s[:'2013/01/01 09:30'], s[:1860])
    for d in ['2013/01/01', '2013/01', '2013']:
        tm.assert_series_equal(s[d:], s)