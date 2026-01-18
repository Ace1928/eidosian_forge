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
def test_pacf2acf_errors():
    pacf = -0.9 ** np.arange(11.0)
    pacf[0] = 1
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf, nlags=20)
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf[1:])
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros(10))
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros((10, 2)))