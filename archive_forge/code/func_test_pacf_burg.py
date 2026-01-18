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
def test_pacf_burg():
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]
    pacf, sigma2 = pacf_burg(y, 10)
    yw_pacf = pacf_yw(y, 10)
    assert_allclose(pacf, yw_pacf, atol=0.0005)
    ye = y - y.mean()
    s2y = ye.dot(ye) / 10000
    pacf[0] = 0
    sigma2_direct = s2y * np.cumprod(1 - pacf ** 2)
    assert_allclose(sigma2, sigma2_direct, atol=0.001)