import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_smoothed_varmax(use_exog, trend):
    endog = np.log(dta[['realgdp', 'cpi']])
    if trend in ['n', 'c']:
        endog = endog.diff().iloc[1:] * 100
    if trend == 'n':
        endog -= endog.mean()
    exog = np.ones(100) if use_exog else None
    if use_exog:
        exog[20:40] = 2.0
    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1.0, 2.0]
    cov_params = [1.0, 0.0, 1.0]
    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params
    x_fit = exog[:50] if use_exog else None
    mod = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x_fit)
    mod['obs_intercept'] = [5, -2.0]
    mod['obs_cov'] = np.array([[1.2, 0.3], [0.3, 3.4]])
    res = mod.smooth(params)
    x_fcast = exog[50:61] if use_exog else None
    p_pred = res.get_prediction(start=0, end=60, information_set='predicted', exog=x_fcast)
    f_pred = res.get_prediction(start=0, end=60, information_set='filtered', exog=x_fcast)
    s_pred = res.get_prediction(start=0, end=60, information_set='smoothed', exog=x_fcast)
    fcast = res.get_forecast(11, exog=x_fcast)
    d = mod['obs_intercept'][:, None]
    Z = mod['design']
    H = mod['obs_cov'][:, :, None]
    desired_s_signal = Z @ res.smoothed_state
    desired_f_signal = Z @ res.filtered_state
    desired_p_signal = Z @ res.predicted_state[..., :-1]
    assert_allclose(s_pred.predicted_mean[:50], (d + desired_s_signal).T)
    assert_allclose(s_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(f_pred.predicted_mean[:50], (d + desired_f_signal).T)
    assert_allclose(f_pred.predicted_mean[50:], fcast.predicted_mean)
    assert_allclose(p_pred.predicted_mean[:50], (d + desired_p_signal).T)
    assert_allclose(p_pred.predicted_mean[50:], fcast.predicted_mean)
    desired_s_signal_cov = Z[None, :, :] @ res.smoothed_state_cov.T @ Z.T[None, :, :]
    desired_f_signal_cov = Z[None, :, :] @ res.filtered_state_cov.T @ Z.T[None, :, :]
    desired_p_signal_cov = Z[None, :, :] @ res.predicted_state_cov[..., :-1].T @ Z.T[None, :, :]
    assert_allclose(s_pred.var_pred_mean[:50], (desired_s_signal_cov.T + H).T)
    assert_allclose(s_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(f_pred.var_pred_mean[:50], (desired_f_signal_cov.T + H).T)
    assert_allclose(f_pred.var_pred_mean[50:], fcast.var_pred_mean)
    assert_allclose(p_pred.var_pred_mean[:50], (desired_p_signal_cov.T + H).T)
    assert_allclose(p_pred.var_pred_mean[50:], fcast.var_pred_mean)
    p_signal = res.get_prediction(start=0, end=60, information_set='predicted', signal_only=True, exog=x_fcast)
    f_signal = res.get_prediction(start=0, end=60, information_set='filtered', signal_only=True, exog=x_fcast)
    s_signal = res.get_prediction(start=0, end=60, information_set='smoothed', signal_only=True, exog=x_fcast)
    fcast_signal = fcast.predicted_mean - d.T
    fcast_signal_cov = (fcast.var_pred_mean.T - H).T
    assert_allclose(s_signal.predicted_mean[:50], desired_s_signal.T)
    assert_allclose(s_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(f_signal.predicted_mean[:50], desired_f_signal.T)
    assert_allclose(f_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(p_signal.predicted_mean[:50], desired_p_signal.T)
    assert_allclose(p_signal.predicted_mean[50:], fcast_signal)
    assert_allclose(s_signal.var_pred_mean[:50], desired_s_signal_cov)
    assert_allclose(s_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(f_signal.var_pred_mean[:50], desired_f_signal_cov)
    assert_allclose(f_signal.var_pred_mean[50:], fcast_signal_cov)
    assert_allclose(p_signal.var_pred_mean[:50], desired_p_signal_cov)
    assert_allclose(p_signal.var_pred_mean[50:], fcast_signal_cov)