import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype,expected_dtype', [(np.int8, np.float64), (np.int16, np.float64), (np.int32, np.float64), (np.int64, np.float64), (np.float32, np.float32), (np.float64, np.float64)])
def test_where_unsafe_upcast(dtype, expected_dtype):
    s = Series(np.arange(10), dtype=dtype)
    values = [2.5, 3.5, 4.5, 5.5, 6.5]
    mask = s < 5
    expected = Series(values + list(range(5, 10)), dtype=expected_dtype)
    warn = None if np.dtype(dtype).kind == np.dtype(expected_dtype).kind == 'f' else FutureWarning
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        s[mask] = values
    tm.assert_series_equal(s, expected)