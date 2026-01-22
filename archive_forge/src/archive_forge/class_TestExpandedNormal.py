import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
class TestExpandedNormal:

    def test_too_few_cumulants(self):
        assert_raises(ValueError, ExpandedNormal, [1])

    def test_coefficients(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ne3 = ExpandedNormal([0.0, 1.0, 1.0])
            assert_allclose(ne3._coef, [1.0, 0.0, 0.0, 1.0 / 6])
            ne4 = ExpandedNormal([0.0, 1.0, 1.0, 1.0])
            assert_allclose(ne4._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 0.0, 1.0 / 72])
            ne5 = ExpandedNormal([0.0, 1.0, 1.0, 1.0, 1.0])
            assert_allclose(ne5._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 1.0 / 120, 1.0 / 72, 1.0 / 144, 0.0, 1.0 / 1296])
            ne33 = ExpandedNormal([0.0, 1.0, 1.0, 0.0])
            assert_allclose(ne33._coef, [1.0, 0.0, 0.0, 1.0 / 6, 0.0, 0.0, 1.0 / 72])

    def test_normal(self):
        ne2 = ExpandedNormal([3, 4])
        x = np.linspace(-2.0, 2.0, 100)
        assert_allclose(ne2.pdf(x), stats.norm.pdf(x, loc=3, scale=2))

    def test_chi2_moments(self):
        N, df = (6, 15)
        cum = [_chi2_cumulant(n + 1, df) for n in range(N)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ne = ExpandedNormal(cum, name='edgw_chi2')
        assert_allclose([_chi2_moment(n, df) for n in range(N)], [ne.moment(n) for n in range(N)])
        check_pdf(ne, arg=(), msg='')
        check_cdf_ppf(ne, arg=(), msg='')
        check_cdf_sf(ne, arg=(), msg='')
        np.random.seed(765456)
        rvs = ne.rvs(size=500)
        check_distribution_rvs(ne, args=(), alpha=0.01, rvs=rvs)

    def test_pdf_no_roots(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            ne = ExpandedNormal([0, 1])
            ne = ExpandedNormal([0, 1, 0.1, 0.1])

    def test_pdf_has_roots(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_raises(RuntimeWarning, ExpandedNormal, [0, 1, 101])