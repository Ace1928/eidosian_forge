import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ufunc', [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    a = pd.array([1.0, 0.2, 3.0, np.nan], dtype='Float64')
    with np.errstate(invalid='ignore'):
        result = ufunc(a)
        expected = pd.array(ufunc(a.astype(float)), dtype='Float64')
    tm.assert_extension_array_equal(result, expected)
    s = pd.Series(a)
    with np.errstate(invalid='ignore'):
        result = ufunc(s)
        expected = pd.Series(ufunc(s.astype(float)), dtype='Float64')
    tm.assert_series_equal(result, expected)