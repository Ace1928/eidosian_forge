import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_predicted_filtered_smoothed_with_nans(use_exog, trend):
    endog = np.zeros(200).reshape(100, 2) * np.nan
    exog = np.ones(100) if use_exog else None
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
    res = mod.smooth(params)
    x_fcast = exog[50:61] if use_exog else None
    p_pred = res.get_prediction(start=0, end=60, information_set='predicted', exog=x_fcast)
    f_pred = res.get_prediction(start=0, end=60, information_set='filtered', exog=x_fcast)
    s_pred = res.get_prediction(start=0, end=60, information_set='smoothed', exog=x_fcast)
    assert_allclose(s_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(s_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(f_pred.predicted_mean, p_pred.predicted_mean)
    assert_allclose(f_pred.var_pred_mean, p_pred.var_pred_mean)
    assert_allclose(p_pred.predicted_mean[:50], res.fittedvalues)
    assert_allclose(p_pred.var_pred_mean[:50].T, res.forecasts_error_cov)
    p_signal = res.get_prediction(start=0, end=60, information_set='predicted', signal_only=True, exog=x_fcast)
    f_signal = res.get_prediction(start=0, end=60, information_set='filtered', signal_only=True, exog=x_fcast)
    s_signal = res.get_prediction(start=0, end=60, information_set='smoothed', signal_only=True, exog=x_fcast)
    assert_allclose(s_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(s_signal.var_pred_mean, p_signal.var_pred_mean)
    assert_allclose(f_signal.predicted_mean, p_signal.predicted_mean)
    assert_allclose(f_signal.var_pred_mean, p_signal.var_pred_mean)
    if use_exog is False and trend == 'n':
        assert_allclose(p_signal.predicted_mean[:50], res.fittedvalues)
        assert_allclose(p_signal.var_pred_mean[:50].T, res.forecasts_error_cov)
    else:
        assert_allclose(p_signal.predicted_mean[:50] + mod['obs_intercept'], res.fittedvalues)
        assert_allclose((p_signal.var_pred_mean[:50] + mod['obs_cov']).T, res.forecasts_error_cov)