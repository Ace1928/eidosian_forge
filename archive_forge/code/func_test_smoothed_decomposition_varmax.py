import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax, dynamic_factor_mq
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [False, True])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_smoothed_decomposition_varmax(use_exog, trend):
    endog = np.array([[0.2, 1.0], [-0.3, -0.5], [0.01, 0.4], [-0.4, 0.1], [0.1, 0.05]])
    endog[0, 0] = np.nan
    endog[1, :] = np.nan
    endog[2, 1] = np.nan
    exog = np.array([2, 5.3, -1, 3.4, 0.0]) if use_exog else None
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
    mod = varmax.VARMAX(endog, order=(1, 0), trend=trend, exog=exog if use_exog else None)
    prior_mean = np.array([-0.4, 0.9])
    prior_cov = np.array([[1.4, 0.3], [0.3, 2.6]])
    mod.ssm.initialize_known(prior_mean, prior_cov)
    res = mod.smooth(params)
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_state')
    css = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    css = css.unstack(level='state_to').values
    ss = np.array(res.states.smoothed)
    assert_allclose(css, ss, atol=1e-12)
    csf = (css.T * mod['design'][:, :, None]).sum(axis=1).T
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)
    assert_allclose(csf, sf, atol=1e-12)
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_signal')
    cs_sig = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    cs_sig = cs_sig.unstack(level='variable_to').values
    assert_allclose(cs_sig, s_sig, atol=1e-12)
    csf = cs_sig + mod['obs_intercept'].T
    assert_allclose(csf, sf)