import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_string_dtype(self):
    ser = pd.Series(['one', 'two', np.nan], dtype='string')
    res = ser.replace({'one': '1', 'two': '2'})
    expected = pd.Series(['1', '2', np.nan], dtype='string')
    tm.assert_series_equal(res, expected)
    ser2 = pd.Series(['A', np.nan], dtype='string')
    res2 = ser2.replace('A', 'B')
    expected2 = pd.Series(['B', np.nan], dtype='string')
    tm.assert_series_equal(res2, expected2)
    ser3 = pd.Series(['A', 'B'], dtype='string')
    res3 = ser3.replace('A', pd.NA)
    expected3 = pd.Series([pd.NA, 'B'], dtype='string')
    tm.assert_series_equal(res3, expected3)