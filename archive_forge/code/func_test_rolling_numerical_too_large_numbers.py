from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_numerical_too_large_numbers():
    dates = date_range('2015-01-01', periods=10, freq='D')
    ds = Series(data=range(10), index=dates, dtype=np.float64)
    ds.iloc[2] = -9e+33
    result = ds.rolling(5).mean()
    expected = Series([np.nan, np.nan, np.nan, np.nan, -1.8e+33, -1.8e+33, -1.8e+33, 5.0, 6.0, 7.0], index=dates)
    tm.assert_series_equal(result, expected)