from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
def test_ols_inefficient(self):
    lag_len = 5
    pacfols = pacf_ols(self.x, nlags=lag_len, efficient=False)
    x = self.x.copy()
    x -= x.mean()
    n = x.shape[0]
    lags = np.zeros((n - 5, 5))
    lead = x[5:]
    direct = np.empty(lag_len + 1)
    direct[0] = 1.0
    for i in range(lag_len):
        lags[:, i] = x[5 - (i + 1):-(i + 1)]
        direct[i + 1] = lstsq(lags[:, :i + 1], lead, rcond=None)[0][-1]
    assert_allclose(pacfols, direct, atol=1e-08)