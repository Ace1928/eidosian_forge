import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
@pytest.mark.parametrize('dtype', [None, object])
def test_setitem_object_typecode(dtype):
    arr = NumpyExtensionArray(np.array(['a', 'b', 'c'], dtype=dtype))
    arr[0] = 't'
    expected = NumpyExtensionArray(np.array(['t', 'b', 'c'], dtype=dtype))
    tm.assert_extension_array_equal(arr, expected)