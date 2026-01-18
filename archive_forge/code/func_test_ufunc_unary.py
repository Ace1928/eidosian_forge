import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
@pytest.mark.parametrize('ufunc', [np.abs, np.negative, np.positive])
def test_ufunc_unary(ufunc):
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))
    result = ufunc(arr)
    expected = NumpyExtensionArray(ufunc(arr._ndarray))
    tm.assert_extension_array_equal(result, expected)
    out = NumpyExtensionArray(np.array([-9.0, -9.0, -9.0]))
    ufunc(arr, out=out)
    tm.assert_extension_array_equal(out, expected)