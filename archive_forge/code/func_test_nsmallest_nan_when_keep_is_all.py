from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_nsmallest_nan_when_keep_is_all(self):
    s = Series([1, 2, 3, 3, 3, None])
    result = s.nsmallest(3, keep='all')
    expected = Series([1.0, 2.0, 3.0, 3.0, 3.0])
    tm.assert_series_equal(result, expected)
    s = Series([1, 2, None, None, None])
    result = s.nsmallest(3, keep='all')
    expected = Series([1, 2, None, None, None])
    tm.assert_series_equal(result, expected)