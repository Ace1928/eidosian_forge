from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window,expected', [('1ns', [1.0, 1.0, 1.0, 1.0]), ('3ns', [2.0, 3.0, 3.0, 2.0])])
def test_rolling_center_nanosecond_resolution(window, closed, expected, frame_or_series):
    index = date_range('2020', periods=4, freq='1ns')
    df = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result, expected)