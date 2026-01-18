import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
@pytest.mark.parametrize('ar_params,ma_params,sigma2', [(np.array([]), np.array([]), 1), (np.array([0.0]), np.array([0.0]), 1), (np.array([0.9]), np.array([]), 1), (np.array([]), np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_regression_with_arma_errors(ar_params, ma_params, sigma2):
    nobs = 100
    eps = np.random.normal(nobs)
    exog = np.c_[np.ones(nobs), np.random.uniform(size=nobs)]
    beta = [5, -0.2]
    endog = np.dot(exog, beta) + eps
    beta_hat = np.squeeze(np.linalg.pinv(exog).dot(endog))
    demeaned = endog - np.dot(exog, beta_hat)
    llf_obs = arma_innovations.arma_loglikeobs(demeaned, ar_params, ma_params, sigma2)
    mod = SARIMAX(endog, exog=exog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[beta_hat, ar_params, ma_params, sigma2])
    assert_allclose(llf_obs, res.llf_obs)