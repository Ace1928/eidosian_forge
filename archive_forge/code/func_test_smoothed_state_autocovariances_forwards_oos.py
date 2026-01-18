import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
@pytest.mark.parametrize('missing', ['all', 'partial', 'mixed', None])
@pytest.mark.parametrize('filter_univariate', [True, False])
@pytest.mark.parametrize('tvp', [True, False])
def test_smoothed_state_autocovariances_forwards_oos(missing, filter_univariate, tvp):
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=5)
    names = ['obs_intercept', 'design', 'obs_cov', 'transition', 'selection', 'state_cov']
    if not tvp:
        mod, res = get_acov_model(missing, filter_univariate, tvp, params=mod_oos.start_params)
    else:
        mod, _ = get_acov_model(missing, filter_univariate, tvp)
        for name in names:
            mod[name] = mod_oos[name, ..., :-5]
        res = mod.ssm.smooth()
    assert_allclose(res_oos.llf, res.llf)
    cov = np.concatenate((res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2).transpose(2, 0, 1)
    desired_acov1 = cov[:, 2:4, :2]
    desired_acov2 = cov[:, 4:6, :2]
    desired_acov3 = cov[:, 6:8, :2]
    extend_kwargs = {}
    if tvp:
        extend_kwargs = {'obs_intercept': mod_oos['obs_intercept', ..., -5:], 'design': mod_oos['design', ..., -5:], 'obs_cov': mod_oos['obs_cov', ..., -5:], 'transition': mod_oos['transition', ..., -5:], 'selection': mod_oos['selection', ..., -5:], 'state_cov': mod_oos['state_cov', ..., -5:]}
    acov1 = res.smoothed_state_autocovariance(-1, end=mod_oos.nobs, extend_kwargs=extend_kwargs).transpose(2, 0, 1)
    assert_equal(acov1.shape, (mod_oos.nobs, mod.k_states, mod.k_states))
    assert_allclose(acov1[:, :2, :2], desired_acov1[1:])
    acov2 = res.smoothed_state_autocovariance(-2, end=mod_oos.nobs - 1, extend_kwargs=extend_kwargs).transpose(2, 0, 1)
    assert_equal(acov2.shape, (mod_oos.nobs - 1, mod.k_states, mod.k_states))
    assert_allclose(acov2[:, :2, :2], desired_acov2[2:])
    acov3 = res.smoothed_state_autocovariance(-3, end=mod_oos.nobs - 2, extend_kwargs=extend_kwargs).transpose(2, 0, 1)
    assert_equal(acov3.shape, (mod_oos.nobs - 2, mod.k_states, mod.k_states))
    assert_allclose(acov3[:, :2, :2], desired_acov3[3:])