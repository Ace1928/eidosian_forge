import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_index_series(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2], 'b': 1.5})
    ser = Series([10, 11])
    df = df.set_index(ser)
    expected = df.index.copy(deep=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)
    else:
        tm.assert_index_equal(df.index, Index([100, 11]))