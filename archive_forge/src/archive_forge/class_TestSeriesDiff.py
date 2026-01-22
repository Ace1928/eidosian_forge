import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesDiff:

    def test_diff_np(self):
        ser = Series(np.arange(5))
        res = np.diff(ser)
        expected = np.array([1, 1, 1, 1])
        tm.assert_numpy_array_equal(res, expected)

    def test_diff_int(self):
        a = 10000000000000000
        b = a + 1
        ser = Series([a, b])
        result = ser.diff()
        assert result[1] == 1

    def test_diff_tz(self):
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        ts.diff()
        result = ts.diff(-1)
        expected = ts - ts.shift(-1)
        tm.assert_series_equal(result, expected)
        result = ts.diff(0)
        expected = ts - ts
        tm.assert_series_equal(result, expected)

    def test_diff_dt64(self):
        ser = Series(date_range('20130102', periods=5))
        result = ser.diff()
        expected = ser - ser.shift(1)
        tm.assert_series_equal(result, expected)
        result = result - result.shift(1)
        expected = expected.diff()
        tm.assert_series_equal(result, expected)

    def test_diff_dt64tz(self):
        ser = Series(date_range('2000-01-01 09:00:00', periods=5, tz='US/Eastern'), name='foo')
        result = ser.diff()
        expected = Series(TimedeltaIndex(['NaT'] + ['1 days'] * 4), name='foo')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input,output,diff', [([False, True, True, False, False], [np.nan, True, False, True, False], 1)])
    def test_diff_bool(self, input, output, diff):
        ser = Series(input)
        result = ser.diff()
        expected = Series(output)
        tm.assert_series_equal(result, expected)

    def test_diff_object_dtype(self):
        ser = Series([False, True, 5.0, np.nan, True, False])
        result = ser.diff()
        expected = ser - ser.shift(1)
        tm.assert_series_equal(result, expected)