from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_sum_all_nan_window_floating_artifacts():
    df = DataFrame([0.002, 0.008, 0.005, np.nan, np.nan, np.nan])
    result = df.rolling(3, min_periods=0).sum()
    expected = DataFrame([0.002, 0.01, 0.015, 0.013, 0.005, 0.0])
    tm.assert_frame_equal(result, expected)