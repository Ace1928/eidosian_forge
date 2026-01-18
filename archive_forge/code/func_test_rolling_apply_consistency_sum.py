import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('f', [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_rolling_apply_consistency_sum(request, all_data, rolling_consistency_cases, center, f):
    window, min_periods = rolling_consistency_cases
    if f is np.sum:
        if not no_nans(all_data) and (not (all_na(all_data) and (not all_data.empty) and (min_periods > 0))):
            request.applymarker(pytest.mark.xfail(reason='np.sum has different behavior with NaNs'))
    rolling_f_result = all_data.rolling(window=window, min_periods=min_periods, center=center).sum()
    rolling_apply_f_result = all_data.rolling(window=window, min_periods=min_periods, center=center).apply(func=f, raw=True)
    tm.assert_equal(rolling_f_result, rolling_apply_f_result)