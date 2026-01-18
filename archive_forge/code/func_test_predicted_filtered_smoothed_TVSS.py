import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
def test_predicted_filtered_smoothed_TVSS(reset_randomstate):
    mod = TVSS(np.zeros((50, 2)))
    mod.ssm.initialize_known([1.2, 0.8], np.eye(2))
    res = mod.smooth([])
    mod_oos = TVSS(np.zeros((11, 2)) * np.nan)
    kwargs = {key: mod_oos[key] for key in ['obs_intercept', 'design', 'obs_cov', 'transition', 'selection', 'state_cov']}
    p_pred = res.get_prediction(start=0, end=60, information_set='predicted', **kwargs)
    f_pred = res.get_prediction(start=0, end=60, information_set='filtered', **kwargs)
    s_pred = res.get_prediction(start=0, end=60, information_set='smoothed', **kwargs)
    p_signal = res.get_prediction(start=0, end=60, information_set='predicted', signal_only=True, **kwargs)
    f_signal = res.get_prediction(start=0, end=60, information_set='filtered', signal_only=True, **kwargs)
    s_signal = res.get_prediction(start=0, end=60, information_set='smoothed', signal_only=True, **kwargs)
    d = mod['obs_intercept'].transpose(1, 0)[:, :, None]
    Z = mod['design'].transpose(2, 0, 1)
    H = mod['obs_cov'].transpose(2, 0, 1)
    fcast = res.get_forecast(11, **kwargs)
    fcast_signal = fcast.predicted_mean - mod_oos['obs_intercept'].T
    fcast_signal_cov = fcast.var_pred_mean - mod_oos['obs_cov'].T
    desired_s_signal = Z @ res.smoothed_state.T[:, :, None]
    desired_f_signal = Z @ res.filtered_state.T[:, :, None]
    desired_p_signal = Z @ res.predicted_state.T[:-1, :, None]
    assert_allclose(s_pred.predicted_mean[:50], (d + desired_s_signal)[..., 0])
    assert_allclose(s_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(f_pred.predicted_mean[:50], (d + desired_f_signal)[..., 0])
    assert_allclose(f_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(p_pred.predicted_mean[:50], (d + desired_p_signal)[..., 0])
    assert_allclose(p_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(s_signal.predicted_mean[:50], desired_s_signal[..., 0])
    assert_allclose(s_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(f_signal.predicted_mean[:50], desired_f_signal[..., 0])
    assert_allclose(f_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(p_signal.predicted_mean[:50], desired_p_signal[..., 0])
    assert_allclose(p_signal.predicted_mean[50:], fcast_signal)
    for t in range(mod.nobs):
        assert_allclose(s_pred.var_pred_mean[t], Z[t] @ res.smoothed_state_cov[..., t] @ Z[t].T + H[t])
        assert_allclose(f_pred.var_pred_mean[t], Z[t] @ res.filtered_state_cov[..., t] @ Z[t].T + H[t])
        assert_allclose(p_pred.var_pred_mean[t], Z[t] @ res.predicted_state_cov[..., t] @ Z[t].T + H[t])
        assert_allclose(s_signal.var_pred_mean[t], Z[t] @ res.smoothed_state_cov[..., t] @ Z[t].T)
        assert_allclose(f_signal.var_pred_mean[t], Z[t] @ res.filtered_state_cov[..., t] @ Z[t].T)
        assert_allclose(p_signal.var_pred_mean[t], Z[t] @ res.predicted_state_cov[..., t] @ Z[t].T)
    assert_allclose(s_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(f_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(p_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(s_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(f_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(p_signal.var_pred_mean[50:], fcast_signal_cov)