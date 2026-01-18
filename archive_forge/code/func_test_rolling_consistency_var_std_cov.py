import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('ddof', [0, 1])
def test_rolling_consistency_var_std_cov(all_data, rolling_consistency_cases, center, ddof):
    window, min_periods = rolling_consistency_cases
    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(ddof=ddof)
    assert not (var_x < 0).any().any()
    std_x = all_data.rolling(window=window, min_periods=min_periods, center=center).std(ddof=ddof)
    assert not (std_x < 0).any().any()
    tm.assert_equal(var_x, std_x * std_x)
    cov_x_x = all_data.rolling(window=window, min_periods=min_periods, center=center).cov(all_data, ddof=ddof)
    assert not (cov_x_x < 0).any().any()
    tm.assert_equal(var_x, cov_x_x)