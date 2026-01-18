import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_like', [True, False])
def test_series_equal_order_mismatch(check_like):
    s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
    s2 = Series([3, 2, 1], index=['c', 'b', 'a'])
    if not check_like:
        with pytest.raises(AssertionError, match='Series.index are different'):
            tm.assert_series_equal(s1, s2, check_like=check_like)
    else:
        _assert_series_equal_both(s1, s2, check_like=check_like)