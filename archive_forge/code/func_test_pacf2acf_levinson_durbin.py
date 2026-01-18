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
def test_pacf2acf_levinson_durbin():
    pacf = -0.9 ** np.arange(11.0)
    pacf[0] = 1
    ar, acf = levinson_durbin_pacf(pacf)
    _, ar_ld, pacf_ld, _, _ = levinson_durbin(acf, 10, isacov=True)
    assert_allclose(ar, ar_ld, atol=1e-08)
    assert_allclose(pacf, pacf_ld, atol=1e-08)
    ar_from_r = [-4.1609, -9.2549, -14.4826, -17.6505, -17.5012, -14.2969, -9.502, -4.9184, -1.7911, -0.3486]
    assert_allclose(ar, ar_from_r, atol=0.0001)