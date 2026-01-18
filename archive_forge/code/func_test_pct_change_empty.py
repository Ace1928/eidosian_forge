import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pct_change_empty():
    ser = Series([], dtype='float64')
    expected = ser.copy()
    result = ser.pct_change(periods=0)
    tm.assert_series_equal(expected, result)