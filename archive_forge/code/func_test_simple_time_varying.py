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
def test_simple_time_varying():
    endog = np.arange(100) * 1.0
    exog = 2 * endog
    mod = sarimax.SARIMAX(endog, exog=exog, order=(0, 0, 0), time_varying_regression=True, mle_regression=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = mod.fit(disp=-1)
    assert_almost_equal(res.params, [0, 0], 5)
    assert_almost_equal(res.filter_results.filtered_state[0][1:], [0.5] * 99, 9)