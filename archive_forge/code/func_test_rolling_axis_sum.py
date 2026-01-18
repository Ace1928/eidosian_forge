from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_axis_sum(axis_frame):
    df = DataFrame(np.ones((10, 20)))
    axis = df._get_axis_number(axis_frame)
    if axis == 0:
        msg = "The 'axis' keyword in DataFrame.rolling"
        expected = DataFrame({i: [np.nan] * 2 + [3.0] * 8 for i in range(20)})
    else:
        msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
        expected = DataFrame([[np.nan] * 2 + [3.0] * 18] * 10)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(3, axis=axis_frame).sum()
    tm.assert_frame_equal(result, expected)