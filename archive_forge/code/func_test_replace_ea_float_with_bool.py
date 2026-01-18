import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_ea_float_with_bool(self):
    ser = pd.Series([0.0], dtype='Float64')
    expected = ser.copy()
    result = ser.replace(False, 1.0)
    tm.assert_series_equal(result, expected)
    ser = pd.Series([False], dtype='boolean')
    expected = ser.copy()
    result = ser.replace(0.0, True)
    tm.assert_series_equal(result, expected)