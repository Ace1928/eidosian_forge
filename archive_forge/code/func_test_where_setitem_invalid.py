import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set ints into string")
def test_where_setitem_invalid():
    msg = lambda x: f'cannot set using a {x} indexer with a different length than the value'
    s = Series(list('abc'))
    with pytest.raises(ValueError, match=msg('slice')):
        s[0:3] = list(range(27))
    s[0:3] = list(range(3))
    expected = Series([0, 1, 2])
    tm.assert_series_equal(s.astype(np.int64), expected)
    s = Series(list('abcdef'))
    with pytest.raises(ValueError, match=msg('slice')):
        s[0:4:2] = list(range(27))
    s = Series(list('abcdef'))
    s[0:4:2] = list(range(2))
    expected = Series([0, 'b', 1, 'd', 'e', 'f'])
    tm.assert_series_equal(s, expected)
    s = Series(list('abcdef'))
    with pytest.raises(ValueError, match=msg('slice')):
        s[:-1] = list(range(27))
    s[-3:-1] = list(range(2))
    expected = Series(['a', 'b', 'c', 0, 1, 'f'])
    tm.assert_series_equal(s, expected)
    s = Series(list('abc'))
    with pytest.raises(ValueError, match=msg('list-like')):
        s[[0, 1, 2]] = list(range(27))
    s = Series(list('abc'))
    with pytest.raises(ValueError, match=msg('list-like')):
        s[[0, 1, 2]] = list(range(2))
    s = Series(list('abc'))
    s[0] = list(range(10))
    expected = Series([list(range(10)), 'b', 'c'])
    tm.assert_series_equal(s, expected)