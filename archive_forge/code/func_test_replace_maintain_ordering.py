import pytest
import pandas as pd
from pandas import Categorical
import pandas._testing as tm
def test_replace_maintain_ordering():
    dtype = pd.CategoricalDtype([0, 1, 2], ordered=True)
    ser = pd.Series([0, 1, 2], dtype=dtype)
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.replace(0, 2)
    expected_dtype = pd.CategoricalDtype([1, 2], ordered=True)
    expected = pd.Series([2, 1, 2], dtype=expected_dtype)
    tm.assert_series_equal(expected, result, check_category_order=True)