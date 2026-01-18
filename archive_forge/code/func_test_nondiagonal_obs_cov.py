from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
def test_nondiagonal_obs_cov(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    res1 = mod.smooth([])
    mod.ssm.filter_univariate = True
    res2 = mod.smooth([])
    atol = 0.002 if PLATFORM_WIN else 1e-05
    rtol = 0.002 if PLATFORM_WIN else 0.0001
    assert_allclose(res1.llf, res2.llf, rtol=rtol, atol=atol)
    assert_allclose(res1.forecasts[0], res2.forecasts[0], rtol=rtol, atol=atol)
    assert_allclose(res1.filtered_state, res2.filtered_state, rtol=rtol, atol=atol)
    assert_allclose(res1.filtered_state_cov, res2.filtered_state_cov, rtol=rtol, atol=atol)
    assert_allclose(res1.smoothed_state, res2.smoothed_state, rtol=rtol, atol=atol)
    assert_allclose(res1.smoothed_state_cov, res2.smoothed_state_cov, rtol=rtol, atol=atol)