import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_smoothed_state_obs_weights_collapsed(reset_randomstate):
    endog = np.zeros((20, 6))
    endog[2, :] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod['obs_intercept'] = np.zeros((6, 1))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    mod.ssm.filter_collapsed = True
    res = mod.smooth([])
    n = mod.nobs
    m = mod.k_states
    p = mod.k_endog
    desired = np.zeros((n, n, m, p)) * np.nan
    for j in range(n):
        for i in range(p):
            if np.isnan(endog[j, i]):
                desired[:, j, :, i] = np.nan
            else:
                y = endog.copy()
                y[j, i] = 1.0
                tmp_mod = mod.clone(y)
                tmp_mod['obs_intercept'] = np.zeros((6, 1))
                tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
                mod.ssm.filter_collapsed = True
                tmp_res = tmp_mod.smooth([])
                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    for j in range(n):
        for ell in range(m):
            tmp_mod = mod.clone(endog)
            tmp_mod['obs_intercept'] = np.zeros((6, 1))
            tmp_mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
            mod.ssm.filter_collapsed = True
            if tmp_mod['state_intercept'].ndim == 1:
                si = tmp_mod['state_intercept']
                tmp_mod['state_intercept'] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod['state_intercept', :, :] = si[:, None]
            tmp_mod['state_intercept', ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()
            desired_state_intercept_weights[:, j, :, ell] = tmp_res.smoothed_state.T - res.smoothed_state.T
    actual, actual_state_intercept_weights, _ = tools.compute_smoothed_state_weights(res)
    assert_allclose(actual, desired, atol=1e-12)
    assert_allclose(actual_state_intercept_weights, desired_state_intercept_weights, atol=1e-12)