import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('ufunc', [np.add, np.subtract])
def test_ufuncs_binary_int(ufunc):
    a = pd.array([1, 2, -3, np.nan])
    result = ufunc(a, a)
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a.astype(float), arr), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a.astype(float)), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(a, 1)
    expected = pd.array(ufunc(a.astype(float), 1), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(1, a)
    expected = pd.array(ufunc(1, a.astype(float)), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)