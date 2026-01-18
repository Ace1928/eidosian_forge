import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
def test_ufunc_with_out(self, dtype):
    arr = pd.array([1, 2, 3], dtype=dtype)
    arr2 = pd.array([1, 2, pd.NA], dtype=dtype)
    mask = arr == arr
    mask2 = arr2 == arr2
    result = np.zeros(3, dtype=bool)
    result |= mask
    assert isinstance(result, np.ndarray)
    assert result.all()
    result = np.zeros(3, dtype=bool)
    msg = "Specify an appropriate 'na_value' for this dtype"
    with pytest.raises(ValueError, match=msg):
        result |= mask2
    res = np.add(arr, arr2)
    expected = pd.array([2, 4, pd.NA], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)
    res = np.add(arr, arr2, out=arr)
    assert res is arr
    tm.assert_extension_array_equal(res, expected)
    tm.assert_extension_array_equal(arr, expected)