from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_scalar_na_logical_ops_corners(self):
    s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])
    msg = 'Cannot perform.+with a dtyped.+array and scalar of type'
    with pytest.raises(TypeError, match=msg):
        s & datetime(2005, 1, 1)
    s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
    s[::2] = np.nan
    expected = Series(True, index=s.index)
    expected[::2] = False
    msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s & list(s)
    tm.assert_series_equal(result, expected)