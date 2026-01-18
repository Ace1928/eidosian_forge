import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ufunc', [np.abs, np.sign])
@pytest.mark.filterwarnings('ignore:invalid value encountered in sign:RuntimeWarning')
def test_ufuncs_single(ufunc):
    a = pd.array([1, 2, -3, np.nan], dtype='Float64')
    result = ufunc(a)
    expected = pd.array(ufunc(a.astype(float)), dtype='Float64')
    tm.assert_extension_array_equal(result, expected)
    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)