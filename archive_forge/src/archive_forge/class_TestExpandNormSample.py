import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
class TestExpandNormSample:

    @classmethod
    def setup_class(cls):
        cls.dist1 = dist1 = stats.norm(1, 2)
        np.random.seed(5999)
        cls.rvs = dist1.rvs(size=200)
        cls.dist2 = NormExpan_gen(cls.rvs, mode='sample')
        cls.scale = 2
        cls.atol_pdf = 0.001

    def test_ks(self):
        stat, pvalue = stats.kstest(self.rvs, self.dist2.cdf)
        assert_array_less(0.25, pvalue)

    def test_mvsk(self):
        mvsk = stats.describe(self.rvs)[-4:]
        assert_allclose(self.dist2.mvsk, mvsk, rtol=1e-12)