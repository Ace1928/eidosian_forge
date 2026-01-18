import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_error():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0
    msg = 'Array conditional must be same shape as self'
    with pytest.raises(ValueError, match=msg):
        s.where(1)
    with pytest.raises(ValueError, match=msg):
        s.where(cond[:3].values, -s)
    s = Series([1, 2])
    s[[True, False]] = [0, 1]
    expected = Series([0, 2])
    tm.assert_series_equal(s, expected)
    msg = 'cannot set using a list-like indexer with a different length than the value'
    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = [0, 2, 3]
    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = []