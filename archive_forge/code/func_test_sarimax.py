import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
@pytest.mark.parametrize('use_exact_diffuse', [False, True])
def test_sarimax(missing, periods, use_exact_diffuse):
    endog = np.array([0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9])
    exog = np.ones_like(endog)
    if missing == 'init':
        endog[0:2] = np.nan
    elif missing == 'mixed':
        endog[2:4] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='t', seasonal_order=(1, 1, 1, 2), exog=exog, use_exact_diffuse=use_exact_diffuse)
    mod.update([0.1, 0.3, 0.5, 0.2, 0.05, -0.1, 1.0])
    check_filter_output(mod, periods, atol=1e-08)
    check_smoother_output(mod, periods)