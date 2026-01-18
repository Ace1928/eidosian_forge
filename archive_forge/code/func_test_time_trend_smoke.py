from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.smoke
def test_time_trend_smoke(index, forecast_index):
    tt = TimeTrend(True, 2)
    tt.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if is_int_index(index) and np.any(np.diff(index) != 1) or (type(index) is pd.Index and max(index) > 2 ** 63 and (forecast_index is None)):
        warn = UserWarning
    with pytest_warns(warn):
        tt.out_of_sample(steps, index, forecast_index)
    str(tt)
    hash(tt)
    assert isinstance(tt.order, int)
    assert isinstance(tt._constant, bool)
    assert TimeTrend.from_string('ctt') == tt
    assert TimeTrend.from_string('ct') != tt
    assert TimeTrend.from_string('t') != tt
    assert TimeTrend.from_string('n') != tt
    assert Seasonality(12) != tt
    tt0 = TimeTrend(False, 0)
    tt0.in_sample(index)
    str(tt0)