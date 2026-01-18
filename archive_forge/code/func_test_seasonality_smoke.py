from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.smoke
def test_seasonality_smoke(index, forecast_index):
    s = Seasonality(12)
    s.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if is_int_index(index) and np.any(np.diff(index) != 1) or (type(index) is pd.Index and max(index) > 2 ** 63 and (forecast_index is None)):
        warn = UserWarning
    with pytest_warns(warn):
        s.out_of_sample(steps, index, forecast_index)
    assert isinstance(s.period, int)
    str(s)
    hash(s)
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) and index.freq:
        s = Seasonality.from_index(index)
        s.in_sample(index)
        s.out_of_sample(steps, index, forecast_index)
        Seasonality.from_index(list(index))