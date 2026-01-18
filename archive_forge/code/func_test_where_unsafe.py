import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_unsafe():
    s = Series(np.arange(10), dtype='int64')
    values = [2.5, 3.5, 4.5, 5.5]
    mask = s > 5
    expected = Series(list(range(6)) + values, dtype='float64')
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        s[mask] = values
    tm.assert_series_equal(s, expected)
    s = Series(np.arange(10), dtype='int64')
    mask = s < 5
    s[mask] = range(2, 7)
    expected = Series(list(range(2, 7)) + list(range(5, 10)), dtype='int64')
    tm.assert_series_equal(s, expected)
    assert s.dtype == expected.dtype
    s = Series(np.arange(10), dtype='int64')
    mask = s > 5
    s[mask] = [0] * 4
    expected = Series([0, 1, 2, 3, 4, 5] + [0] * 4, dtype='int64')
    tm.assert_series_equal(s, expected)
    s = Series(np.arange(10))
    mask = s > 5
    msg = 'cannot set using a list-like indexer with a different length than the value'
    with pytest.raises(ValueError, match=msg):
        s[mask] = [5, 4, 3, 2, 1]
    with pytest.raises(ValueError, match=msg):
        s[mask] = [0] * 5
    s = Series([1, 2, 3, 4])
    result = s.where(s > 2, np.nan)
    expected = Series([np.nan, np.nan, 3, 4])
    tm.assert_series_equal(result, expected)
    s = Series(range(10)).astype(float)
    s[8] = None
    result = s[8]
    assert isna(result)
    s = Series(range(10)).astype(float)
    s[s > 8] = None
    result = s[isna(s)]
    expected = Series(np.nan, index=[9])
    tm.assert_series_equal(result, expected)