import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_by_key(self):
    df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})
    result = df.sort_values('a', key=lambda x: -x)
    expected = df.iloc[[1, 3, 4, 0, 2, 5]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=['a', 'b'], key=lambda x: -x)
    expected = df.iloc[[3, 1, 4, 0, 2, 5]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=['a', 'b'], key=lambda x: -x, ascending=False)
    expected = df.iloc[[0, 4, 1, 3, 2, 5]]
    tm.assert_frame_equal(result, expected)