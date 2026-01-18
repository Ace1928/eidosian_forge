import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ufunc', [np.logical_not])
def test_ufuncs_unary(ufunc):
    a = pd.array([True, False, None], dtype='boolean')
    result = ufunc(a)
    expected = pd.array(ufunc(a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)
    ser = pd.Series(a)
    result = ufunc(ser)
    expected = pd.Series(ufunc(a._data), dtype='boolean')
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)