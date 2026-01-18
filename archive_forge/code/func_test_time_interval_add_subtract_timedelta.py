from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['__add__', '__sub__'])
@pytest.mark.parametrize('interval', [Interval(Timestamp('2017-01-01 00:00:00'), Timestamp('2018-01-01 00:00:00')), Interval(Timedelta(days=7), Timedelta(days=14))])
@pytest.mark.parametrize('delta', [Timedelta(days=7), timedelta(7), np.timedelta64(7, 'D')])
def test_time_interval_add_subtract_timedelta(self, interval, delta, method):
    result = getattr(interval, method)(delta)
    left = getattr(interval.left, method)(delta)
    right = getattr(interval.right, method)(delta)
    expected = Interval(left, right)
    assert result == expected