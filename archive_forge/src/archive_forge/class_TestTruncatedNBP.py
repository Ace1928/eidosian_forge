import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestTruncatedNBP:
    """
    Test Truncated Poisson distribution
    """

    def test_pmf_zero(self):
        n, p = truncatednegbin.convert_params(5, 0.1, 2)
        nb_pmf = nbinom.pmf(1, n, p) / nbinom.sf(0, n, p)
        tnb_pmf = truncatednegbin.pmf(1, 5, 0.1, 2, 0)
        assert_allclose(nb_pmf, tnb_pmf, rtol=1e-05)

    def test_logpmf_zero(self):
        n, p = truncatednegbin.convert_params(5, 1, 2)
        nb_logpmf = nbinom.logpmf(1, n, p) - np.log(nbinom.sf(0, n, p))
        tnb_logpmf = truncatednegbin.logpmf(1, 5, 1, 2, 0)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=0.01, atol=0.01)

    def test_pmf(self):
        n, p = truncatednegbin.convert_params(2, 0.5, 2)
        nb_logpmf = nbinom.pmf(6, n, p) / nbinom.sf(5, n, p)
        tnb_pmf = truncatednegbin.pmf(6, 2, 0.5, 2, 5)
        assert_allclose(nb_logpmf, tnb_pmf, rtol=1e-07)
        tnb_pmf = truncatednegbin.pmf(5, 2, 0.5, 2, 5)
        assert_equal(tnb_pmf, 0)

    def test_logpmf(self):
        n, p = truncatednegbin.convert_params(5, 0.1, 2)
        nb_logpmf = nbinom.logpmf(6, n, p) - np.log(nbinom.sf(5, n, p))
        tnb_logpmf = truncatednegbin.logpmf(6, 5, 0.1, 2, 5)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=1e-07)
        tnb_logpmf = truncatednegbin.logpmf(5, 5, 0.1, 2, 5)
        assert np.isneginf(tnb_logpmf)