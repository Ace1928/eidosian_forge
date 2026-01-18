import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ufunc', [np.add, np.logical_or, np.logical_and, np.logical_xor])
def test_ufuncs_binary(ufunc):
    a = pd.array([True, False, None], dtype='boolean')
    result = ufunc(a, a)
    expected = pd.array(ufunc(a._data, a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    s = pd.Series(a)
    result = ufunc(s, a)
    expected = pd.Series(ufunc(a._data, a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)
    arr = np.array([True, True, False])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a._data, arr), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(a, True)
    expected = pd.array(ufunc(a._data, True), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    result = ufunc(True, a)
    expected = pd.array(ufunc(True, a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    msg = 'operand type\\(s\\) all returned NotImplemented from __array_ufunc__'
    with pytest.raises(TypeError, match=msg):
        ufunc(a, 'test')