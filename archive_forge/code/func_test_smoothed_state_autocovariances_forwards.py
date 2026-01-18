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
def test_smoothed_state_autocovariances_forwards(missing, filter_univariate, tvp):
    """
    Test for Cov(t, t + lag)
    """
    mod_oos, res_oos = get_acov_model(missing, filter_univariate, tvp, oos=3)
    names = ['obs_intercept', 'design', 'obs_cov', 'transition', 'selection', 'state_cov']
    if not tvp:
        mod, res = get_acov_model(missing, filter_univariate, tvp, params=mod_oos.start_params)
    else:
        mod, _ = get_acov_model(missing, filter_univariate, tvp)
        for name in names:
            mod[name] = mod_oos[name, ..., :-3]
        res = mod.ssm.smooth()
    extend_kwargs1 = {}
    extend_kwargs2 = {}
    if tvp:
        keys = ['obs_intercept', 'design', 'obs_cov', 'transition', 'selection', 'state_cov']
        for key in keys:
            extend_kwargs1[key] = mod_oos[key, ..., -3:-2]
            extend_kwargs2[key] = mod_oos[key, ..., -3:-1]
    assert_allclose(res_oos.llf, res.llf)
    cov = res.smoothed_state_cov.transpose(2, 0, 1)
    desired_acov1 = cov[:, 2:4, :2]
    desired_acov2 = cov[:, 4:6, :2]
    desired_acov3 = cov[:, 6:8, :2]
    oos_cov = np.concatenate((res_oos.smoothed_state_cov, res_oos.predicted_state_cov[..., -1:]), axis=2).transpose(2, 0, 1)
    acov1 = res.smoothed_state_autocovariance(-1).transpose(2, 0, 1)
    assert_allclose(acov1[:-1, :2, :2], desired_acov1[1:])
    assert_allclose(acov1[-2:, :2, :2], oos_cov[-5:-3, 2:4, :2])
    acov2 = res.smoothed_state_autocovariance(-2, extend_kwargs=extend_kwargs1).transpose(2, 0, 1)
    assert_allclose(acov2[:-2, :2, :2], desired_acov2[2:])
    assert_allclose(acov2[-2:, :2, :2], oos_cov[-4:-2, 4:6, :2])
    acov3 = res.smoothed_state_autocovariance(-3, extend_kwargs=extend_kwargs2).transpose(2, 0, 1)
    assert_allclose(acov3[:-3, :2, :2], desired_acov3[3:])
    assert_allclose(acov3[-3:, :2, :2], oos_cov[-4:-1, 6:8, :2])
    acov1 = res.smoothed_state_autocovariance(-1, t=mod.nobs, extend_kwargs=extend_kwargs1)
    assert_allclose(acov1[:2, :2], oos_cov[-3, 2:4, :2])
    acov1 = res.smoothed_state_autocovariance(-1, t=0)
    assert_allclose(acov1[:2, :2], desired_acov1[0 + 1])
    acov1 = res.smoothed_state_autocovariance(-1, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov1[:, :2, :2], desired_acov1[8 + 1:9 + 1])
    acov2 = res.smoothed_state_autocovariance(-2, t=mod.nobs, extend_kwargs=extend_kwargs2)
    assert_allclose(acov2[:2, :2], oos_cov[-2, 4:6, :2])
    acov2 = res.smoothed_state_autocovariance(-2, t=mod.nobs - 1, extend_kwargs=extend_kwargs1)
    assert_allclose(acov2[:2, :2], oos_cov[-3, 4:6, :2])
    acov2 = res.smoothed_state_autocovariance(-2, t=0)
    assert_allclose(acov2[:2, :2], desired_acov2[0 + 2])
    acov2 = res.smoothed_state_autocovariance(-2, start=8, end=9).transpose(2, 0, 1)
    assert_allclose(acov2[:, :2, :2], desired_acov2[8 + 2:9 + 2])