import numpy as np
from pandas import (
import pandas._testing as tm
class TestIsna:

    def test_isna_period_dtype(self):
        ser = Series([Period('2011-01', freq='M'), Period('NaT', freq='M')])
        expected = Series([False, True])
        result = ser.isna()
        tm.assert_series_equal(result, expected)
        result = ser.notna()
        tm.assert_series_equal(result, ~expected)

    def test_isna(self):
        ser = Series([0, 5.4, 3, np.nan, -0.001])
        expected = Series([False, False, False, True, False])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)
        ser = Series(['hi', '', np.nan])
        expected = Series([False, False, True])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)