import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_round_ea_boolean(self):
    ser = Series([True, False], dtype='boolean')
    expected = ser.copy()
    result = ser.round(2)
    tm.assert_series_equal(result, expected)
    result.iloc[0] = False
    tm.assert_series_equal(ser, expected)