import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_to_numpy_returns_pdna_default(dtype):
    arr = pd.array(['a', pd.NA, 'b'], dtype=dtype)
    result = np.array(arr)
    expected = np.array(['a', na_val(dtype), 'b'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)