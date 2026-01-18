import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('diffuse', [0, 1, 4])
@pytest.mark.parametrize('univariate', [False, True])
def test_smoothed_state_obs_weights_TVSS(univariate, diffuse, reset_randomstate):
    endog = np.zeros((10, 3))
    if diffuse == 4:
        endog[:3] = np.nan
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    prior_mean = np.array([1.2, 0.8])
    prior_cov = np.eye(2)
    if not diffuse:
        mod.ssm.initialize_known(prior_mean, prior_cov)
    if univariate:
        mod.ssm.filter_univariate = True
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
                if not diffuse:
                    tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
                if univariate:
                    tmp_mod.ssm.filter_univariate = True
                tmp_res = tmp_mod.smooth([])
                desired[:, j, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    desired_state_intercept_weights = np.zeros((n, n, m, m)) * np.nan
    for j in range(n):
        for ell in range(m):
            tmp_mod = mod.clone(endog)
            if not diffuse:
                tmp_mod.ssm.initialize_known(prior_mean, prior_cov)
            if univariate:
                tmp_mod.ssm.filter_univariate = True
            if tmp_mod['state_intercept'].ndim == 1:
                si = tmp_mod['state_intercept']
                tmp_mod['state_intercept'] = np.zeros((mod.k_states, mod.nobs))
                tmp_mod['state_intercept', :, :] = si[:, None]
            tmp_mod['state_intercept', ell, j] += 1.0
            tmp_res = tmp_mod.ssm.smooth()
            desired_state_intercept_weights[:, j, :, ell] = tmp_res.smoothed_state.T - res.smoothed_state.T
    desired_prior_weights = np.zeros((n, m, m)) * np.nan
    if not diffuse:
        for i in range(m):
            a = prior_mean.copy()
            a[i] += 1
            tmp_mod = mod.clone(endog)
            tmp_mod.ssm.initialize_known(a, prior_cov)
            tmp_res = tmp_mod.smooth([])
            desired_prior_weights[:, :, i] = tmp_res.smoothed_state.T - res.smoothed_state.T
    if not diffuse:
        mod.ssm.initialize_known(prior_mean, prior_cov)
    actual, actual_state_intercept_weights, actual_prior_weights = tools.compute_smoothed_state_weights(res)
    d = res.nobs_diffuse
    assert_equal(d, diffuse)
    if diffuse:
        assert_allclose(actual[:d], np.nan, atol=1e-12)
        assert_allclose(actual[:, :d], np.nan, atol=1e-12)
        assert_allclose(actual_state_intercept_weights[:d], np.nan)
        assert_allclose(actual_state_intercept_weights[:, :d], np.nan)
        assert_allclose(actual_prior_weights, np.nan)
    else:
        assert_allclose(actual_prior_weights, desired_prior_weights, atol=1e-12)
        contribution_prior = np.nansum(actual_prior_weights * prior_mean[None, None, :], axis=2)
        contribution_endog = np.nansum(actual * (endog - mod['obs_intercept'].T)[None, :, None, :], axis=(1, 3))
        computed_smoothed_state = contribution_prior + contribution_endog
        assert_allclose(computed_smoothed_state, res.smoothed_state.T)
    assert_allclose(actual[d:, d:], desired[d:, d:], atol=1e-12)
    assert_allclose(actual_state_intercept_weights[d:, d:], desired_state_intercept_weights[d:, d:], atol=1e-12)