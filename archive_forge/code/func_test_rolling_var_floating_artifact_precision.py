from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_var_floating_artifact_precision():
    s = Series([7, 5, 5, 5])
    result = s.rolling(3).var()
    expected = Series([np.nan, np.nan, 4 / 3, 0])
    tm.assert_series_equal(result, expected, atol=1e-15, rtol=1e-15)
    tm.assert_series_equal(result == 0, expected == 0)