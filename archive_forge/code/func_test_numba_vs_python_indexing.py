import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_numba_vs_python_indexing():
    frame = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7.0, 8.0, 9.0]}, index=Index(['A', 'B', 'C']))
    row_func = lambda x: x['c']
    result = frame.apply(row_func, engine='numba', axis=1)
    expected = frame.apply(row_func, engine='python', axis=1)
    tm.assert_series_equal(result, expected)
    col_func = lambda x: x['A']
    result = frame.apply(col_func, engine='numba', axis=0)
    expected = frame.apply(col_func, engine='python', axis=0)
    tm.assert_series_equal(result, expected)