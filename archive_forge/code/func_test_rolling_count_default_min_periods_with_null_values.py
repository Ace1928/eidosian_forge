from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_count_default_min_periods_with_null_values(frame_or_series):
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0]
    result = frame_or_series(values).rolling(3, min_periods=0).count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)