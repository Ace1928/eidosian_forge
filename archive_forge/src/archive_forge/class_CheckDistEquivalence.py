import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
class CheckDistEquivalence:

    def test_cdf(self):
        cdftr = self.dist.cdf(xx, *self.trargs, **self.trkwds)
        sfctr = 1 - self.dist.sf(xx, *self.trargs, **self.trkwds)
        cdfst = self.statsdist.cdf(xx, *self.stargs, **self.stkwds)
        assert_almost_equal(cdfst, cdftr, 14)
        assert_almost_equal(cdfst, sfctr, 14)

    def test_pdf(self):
        pdftr = self.dist.pdf(xx, *self.trargs, **self.trkwds)
        pdfst = self.statsdist.pdf(xx, *self.stargs, **self.stkwds)
        assert_almost_equal(pdfst, pdftr, 13)

    def test_ppf(self):
        ppftr = self.dist.ppf(ppfq, *self.trargs, **self.trkwds)
        ppfst = self.statsdist.ppf(ppfq, *self.stargs, **self.stkwds)
        assert_almost_equal(ppfst, ppftr, 13)

    def test_rvs(self):
        rvs = self.dist.rvs(*self.trargs, **{'size': 100})
        mean_s = rvs.mean(0)
        mean_d, var_d = self.dist.stats(*self.trargs, **{'moments': 'mv'})
        if np.any(np.abs(mean_d) < 1):
            assert_almost_equal(mean_d, mean_s, 1)
        else:
            assert_almost_equal(mean_s / mean_d, 1.0, 0)

    def test_stats(self):
        trkwds = {'moments': 'mvsk'}
        trkwds.update(self.stkwds)
        stkwds = {'moments': 'mvsk'}
        stkwds.update(self.stkwds)
        mvsktr = np.array(self.dist.stats(*self.trargs, **trkwds))
        mvskst = np.array(self.statsdist.stats(*self.stargs, **stkwds))
        assert_almost_equal(mvskst[:2], mvsktr[:2], 8)
        if np.any(np.abs(mvskst[2:]) < 1):
            assert_almost_equal(mvskst[2:], mvsktr[2:], 1)
        else:
            assert_almost_equal(mvskst[2:] / mvsktr[2:], np.ones(2), 0)