import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_sarimax_forecast_exog_trend(reset_randomstate):
    y = np.zeros(10)
    x = np.zeros(10)
    mod = sarimax.SARIMAX(endog=y, exog=x, order=(1, 0, 0), trend='c')
    res = mod.smooth([0.2, 0.4, 0.5, 1.0])
    assert_allclose(res.forecast(1, exog=1), 0.2 + 0.4)
    assert_allclose(res.forecast(2, exog=[1.0, 1.0]), 0.2 + 0.4, 0.2 + 0.4 + 0.5)