import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_setitem_with_scalar_string(dtype):
    arr = pd.array(['a', 'c'], dtype=dtype)
    arr[0] = 'd'
    expected = pd.array(['d', 'c'], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)