import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_rolling_consistency_mean(all_data, rolling_consistency_cases, center):
    window, min_periods = rolling_consistency_cases
    result = all_data.rolling(window=window, min_periods=min_periods, center=center).mean()
    expected = all_data.rolling(window=window, min_periods=min_periods, center=center).sum().divide(all_data.rolling(window=window, min_periods=min_periods, center=center).count())
    tm.assert_equal(result, expected.astype('float64'))