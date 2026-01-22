import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestZIGeneralizedPoisson:

    def test_pmf_zero(self):
        gp_pmf = genpoisson_p.pmf(3, 2, 1, 1)
        zigp_pmf = zigenpoisson.pmf(3, 2, 1, 1, 0)
        assert_allclose(gp_pmf, zigp_pmf, rtol=1e-12)

    def test_logpmf_zero(self):
        gp_logpmf = genpoisson_p.logpmf(7, 3, 1, 1)
        zigp_logpmf = zigenpoisson.logpmf(7, 3, 1, 1, 0)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=1e-12)

    def test_pmf(self):
        gp_pmf = genpoisson_p.pmf(3, 2, 2, 2)
        zigp_pmf = zigenpoisson.pmf(3, 2, 2, 2, 0.1)
        assert_allclose(gp_pmf, zigp_pmf, rtol=0.05, atol=0.05)

    def test_logpmf(self):
        gp_logpmf = genpoisson_p.logpmf(2, 3, 0, 2)
        zigp_logpmf = zigenpoisson.logpmf(2, 3, 0, 2, 0.1)
        assert_allclose(gp_logpmf, zigp_logpmf, rtol=0.05, atol=0.05)

    def test_mean_var(self):
        m = np.array([1, 5, 10])
        poisson_mean, poisson_var = (poisson.mean(m), poisson.var(m))
        zigenpoisson_mean = zigenpoisson.mean(m, 0, 1, 0)
        zigenpoisson_var = zigenpoisson.var(m, 0.0, 1, 0)
        assert_allclose(poisson_mean, zigenpoisson_mean, rtol=1e-10)
        assert_allclose(poisson_var, zigenpoisson_var, rtol=1e-10)