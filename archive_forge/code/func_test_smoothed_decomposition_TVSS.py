import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax, dynamic_factor_mq
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('univariate', [False, True])
def test_smoothed_decomposition_TVSS(univariate, reset_randomstate):
    endog = np.zeros((10, 3))
    endog[6, 0] = np.nan
    endog[7, :] = np.nan
    endog[8, 1] = np.nan
    mod = TVSS(endog)
    mod['state_intercept'] = np.random.normal(size=(mod.k_states, mod.nobs))
    prior_mean = np.array([1.2, 0.8])
    prior_cov = np.eye(2)
    mod.ssm.initialize_known(prior_mean, prior_cov)
    if univariate:
        mod.ssm.filter_univariate = True
    res = mod.smooth([])
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_state')
    css = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    css = css.unstack(level='state_to')[mod.state_names].values
    ss = np.array(res.states.smoothed)
    assert_allclose(css, ss, atol=1e-12)
    cs_sig = (css.T * mod['design']).sum(axis=1).T
    csf = cs_sig + mod['obs_intercept'].T
    s_sig = res.predict(information_set='smoothed', signal_only=True)
    sf = res.predict(information_set='smoothed', signal_only=False)
    assert_allclose(cs_sig, s_sig, atol=1e-12)
    assert_allclose(csf, sf, atol=1e-12)
    cd, coi, csi, cp = res.get_smoothed_decomposition(decomposition_of='smoothed_signal')
    cs_sig = (cd + coi).sum(axis=1) + csi.sum(axis=1) + cp.sum(axis=1)
    cs_sig = cs_sig.unstack(level='variable_to')[mod.endog_names].values
    assert_allclose(cs_sig, s_sig, atol=1e-12)
    csf = cs_sig + mod['obs_intercept'].T
    assert_allclose(csf, sf)