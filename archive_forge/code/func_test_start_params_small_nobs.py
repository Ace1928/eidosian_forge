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
def test_start_params_small_nobs():
    endog = np.log(realgdp_results['value']).diff()[1:].values
    mod = sarimax.SARIMAX(endog[:4], order=(4, 0, 0))
    match = 'Too few observations to estimate starting parameters for ARMA and trend.'
    with pytest.warns(UserWarning, match=match):
        start_params = mod.start_params
        assert_allclose(start_params, [0, 0, 0, 0, np.var(endog[:4])])
    mod = sarimax.SARIMAX(endog[:4], order=(0, 0, 0), seasonal_order=(1, 0, 0, 4))
    match = 'Too few observations to estimate starting parameters for seasonal ARMA.'
    with pytest.warns(UserWarning, match=match):
        start_params = mod.start_params
        assert_allclose(start_params, [0, np.var(endog[:4])])