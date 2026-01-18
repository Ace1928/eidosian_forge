from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_bug_mixed(self, any_signed_int_numpy_dtype):
    df = DataFrame({'a': np.array([1, 2, 3, 4], dtype=any_signed_int_numpy_dtype), 'b': np.array([4.0, 3.0, 2.0, 1.0], dtype='float64')})
    expected = DataFrame({'a': [-1, -1, 3, 4], 'b': [4.0, 3.0, -1, -1]}).astype({'a': any_signed_int_numpy_dtype, 'b': 'float64'})
    result = df.where(df > 2, -1)
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    return_value = result.where(result > 2, -1, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)