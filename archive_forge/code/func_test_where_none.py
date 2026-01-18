from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_none(self):
    df = DataFrame({'series': Series(range(10))}).astype(float)
    df[df > 7] = None
    expected = DataFrame({'series': Series([0, 1, 2, 3, 4, 5, 6, 7, np.nan, np.nan])})
    tm.assert_frame_equal(df, expected)
    df = DataFrame([{'A': 1, 'B': np.nan, 'C': 'Test'}, {'A': np.nan, 'B': 'Test', 'C': np.nan}])
    orig = df.copy()
    mask = ~isna(df)
    df.where(mask, None, inplace=True)
    expected = DataFrame({'A': [1.0, np.nan], 'B': [None, 'Test'], 'C': ['Test', None]})
    tm.assert_frame_equal(df, expected)
    df = orig.copy()
    df[~mask] = None
    tm.assert_frame_equal(df, expected)