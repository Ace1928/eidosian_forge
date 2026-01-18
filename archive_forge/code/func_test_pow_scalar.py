import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_pow_scalar():
    a = pd.array([-1, 0, 1, None, 2], dtype='Int64')
    result = a ** 0
    expected = pd.array([1, 1, 1, 1, 1], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = a ** 1
    expected = pd.array([-1, 0, 1, None, 2], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = a ** pd.NA
    expected = pd.array([None, None, 1, None, None], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = a ** np.nan
    expected = FloatingArray(np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype='float64'), np.array([False, False, False, True, False]))
    tm.assert_extension_array_equal(result, expected)
    a = a[1:]
    result = 0 ** a
    expected = pd.array([1, 0, None, 0], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = 1 ** a
    expected = pd.array([1, 1, 1, 1], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = pd.NA ** a
    expected = pd.array([1, None, None, None], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = np.nan ** a
    expected = FloatingArray(np.array([1, np.nan, np.nan, np.nan], dtype='float64'), np.array([False, False, True, False]))
    tm.assert_extension_array_equal(result, expected)