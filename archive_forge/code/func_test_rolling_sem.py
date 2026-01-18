from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_sem(frame_or_series):
    obj = frame_or_series([0, 1, 2])
    result = obj.rolling(2, min_periods=1).sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.7071067811865476] * 2)
    tm.assert_series_equal(result, expected)