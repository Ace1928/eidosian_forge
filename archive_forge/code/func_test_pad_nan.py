from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_pad_nan(self):
    x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'], dtype=float)
    return_value = x.fillna(method='pad', inplace=True)
    assert return_value is None
    expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ['z', 'a', 'b', 'c', 'd'], dtype=float)
    tm.assert_series_equal(x[1:], expected[1:])
    assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])