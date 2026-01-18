from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
@pytest.mark.parametrize('method', ['estimated', 'heuristic'])
def test_seasonal_order(reset_randomstate, method):
    seasonal = np.arange(12.0)
    time_series = np.array(list(seasonal) * 100)
    res = ETSModel(time_series, seasonal='add', seasonal_periods=12, initialization_method=method).fit()
    assert_allclose(res.initial_seasonal + res.initial_level, seasonal, atol=0.0001, rtol=0.0001)
    assert res.mae < 1e-06