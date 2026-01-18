from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_zero_window():
    s = Series(range(1))
    result = s.rolling(0).min()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)