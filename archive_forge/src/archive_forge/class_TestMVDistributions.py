import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
class TestMVDistributions:

    @classmethod
    def setup_class(cls):
        covx = np.array([[1.0, 0.5], [0.5, 1.0]])
        mu3 = [-1, 0.0, 2.0]
        cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
        cls.mu3 = mu3
        cls.cov3 = cov3
        mvn3 = MVNormal(mu3, cov3)
        mvn3c = MVNormal(np.array([0, 0, 0]), cov3)
        cls.mvn3 = mvn3
        cls.mvn3c = mvn3c

    def test_mvn_pdf(self):
        cov3 = self.cov3
        mvn3 = self.mvn3
        r_val = [-7.667977543898155, -6.917977543898155, -5.167977543898155]
        assert_allclose(mvn3.logpdf(cov3), r_val, rtol=1e-13)
        r_val = [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
        assert_allclose(mvn3.pdf(cov3), r_val, rtol=1e-13)
        mvn3b = MVNormal(np.array([0, 0, 0]), cov3)
        r_val = [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]
        assert_allclose(mvn3b.pdf(cov3), r_val, rtol=1e-13)

    def test_mvt_pdf(self, reset_randomstate):
        cov3 = self.cov3
        mu3 = self.mu3
        mvt = MVT((0, 0), 1, 5)
        assert_almost_equal(mvt.logpdf(np.array([0.0, 0.0])), -1.837877066409345, decimal=15)
        assert_almost_equal(mvt.pdf(np.array([0.0, 0.0])), 0.1591549430918953, decimal=15)
        mvt.logpdf(np.array([1.0, 1.0])) - -3.01552989458359
        mvt1 = MVT((0, 0), 1, 1)
        mvt1.logpdf(np.array([1.0, 1.0])) - -3.48579549941151
        rvs = mvt.rvs(100000)
        assert_almost_equal(np.cov(rvs, rowvar=False), mvt.cov, decimal=1)
        mvt31 = MVT(mu3, cov3, 1)
        assert_almost_equal(mvt31.pdf(cov3), [0.0007276818698165781, 0.0009980625182293658, 0.0027661422056214652], decimal=17)
        mvt = MVT(mu3, cov3, 3)
        assert_almost_equal(mvt.pdf(cov3), [0.00086377742424741, 0.001277510788307594, 0.004156314279452241], decimal=17)