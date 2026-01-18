import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax, dynamic_factor_mq
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
@pytest.mark.parametrize('concentrate_scale', [False, True])
@pytest.mark.parametrize('measurement_error', [False, True])
def test_smoothed_decomposition_sarimax(use_exog, trend, concentrate_scale, measurement_error):
    endog = np.array([[0.2, np.nan, 1.2, -0.3, -1.5]]).T
    exog = np.array([2, 5.3, -1, 3.4, 0.0]) if use_exog else None
    trend_params = [0.1]
    ar_params = [0.5]
    exog_params = [1.4]
    meas_err_params = [1.2]
    cov_params = [0.8]
    params = []
    if trend in ['c', 't']:
        params += trend_params
    if use_exog:
        params += exog_params
    params += ar_params
    if measurement_error:
        params += meas_err_params
    if not concentrate_scale:
        params += cov_params
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend=trend, exog=exog if use_exog else None, concentrate_scale=concentrate_scale, measurement_error=measurement_error)
    prior_mean = np.array([-0.4])
    prior_cov = np.eye(1) * 1.2
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_state')
    css = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    css = css.unstack(level='state_to').values
    ss = np.array(res.states.smoothed)
    assert_allclose(css, ss, atol=1e-12)
    csf = ((css.T * mod['design'][:, :, None]).sum(axis=1) + mod['obs_intercept']).T
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)
    assert_allclose(csf[:, 0], sf)
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_signal')
    cs_sig = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    cs_sig = cs_sig.unstack(level='variable_to').values
    assert_allclose(cs_sig[:, 0], s_sig, atol=1e-12)
    csf = cs_sig + mod['obs_intercept'].T
    assert_allclose(csf[:, 0], sf)