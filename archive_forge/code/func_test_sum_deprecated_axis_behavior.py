import numpy as np
from pandas import (
import pandas._testing as tm
def test_sum_deprecated_axis_behavior(self):
    arr = np.random.default_rng(2).standard_normal((4, 3))
    df = DataFrame(arr)
    msg = 'The behavior of DataFrame.sum with axis=None is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        res = np.sum(df)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.sum(axis=None)
    tm.assert_series_equal(res, expected)