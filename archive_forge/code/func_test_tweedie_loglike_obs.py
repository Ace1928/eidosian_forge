import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
import statsmodels.genmod.families as F
from statsmodels.genmod.families.family import Tweedie
import statsmodels.genmod.families.links as L
@pytest.mark.skipif(SP_LT_17, reason='Scipy too old, function not available')
@pytest.mark.parametrize('power', (1.1, 1.5, 1.9))
def test_tweedie_loglike_obs(power):
    """Test that Tweedie loglike is normalized to 1."""
    tweedie = Tweedie(var_power=power, eql=False)
    mu = 2.0
    scale = 2.9

    def pdf(y):
        return np.squeeze(np.exp(tweedie.loglike_obs(endog=y, mu=mu, scale=scale)))
    assert_allclose(pdf(0) + integrate.quad(pdf, 0, 100.0)[0], 1, atol=0.0001)