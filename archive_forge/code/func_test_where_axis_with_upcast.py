from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_axis_with_upcast(self):
    df = DataFrame([[1, 2], [3, 4]], dtype='int64')
    mask = DataFrame([[False, False], [False, False]])
    ser = Series([0, np.nan])
    expected = DataFrame([[0, 0], [np.nan, np.nan]], dtype='float64')
    result = df.where(mask, ser, axis='index')
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        return_value = result.where(mask, ser, axis='index', inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)
    expected = DataFrame([[0, np.nan], [0, np.nan]])
    result = df.where(mask, ser, axis='columns')
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({0: np.array([0, 0], dtype='int64'), 1: np.array([np.nan, np.nan], dtype='float64')})
    result = df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        return_value = result.where(mask, ser, axis='columns', inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)