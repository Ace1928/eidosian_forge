import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_nan_with_inf(self):
    ser = pd.Series([np.nan, 0, np.inf])
    tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
    ser = pd.Series([np.nan, 0, 'foo', 'bar', np.inf, None, pd.NaT])
    tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
    filled = ser.copy()
    filled[4] = 0
    tm.assert_series_equal(ser.replace(np.inf, 0), filled)