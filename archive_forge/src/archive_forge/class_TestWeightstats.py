import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestWeightstats:

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = (20, 20)
        m1, m2 = (1, 1.2)
        x1 = m1 + np.random.randn(n1)
        x2 = m2 + np.random.randn(n2)
        x1_2d = m1 + np.random.randn(n1, 3)
        x2_2d = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        cls.x1, cls.x2 = (x1, x2)
        cls.w1, cls.w2 = (w1, w2)
        cls.x1_2d, cls.x2_2d = (x1_2d, x2_2d)

    def test_weightstats_1(self):
        x1, x2 = (self.x1, self.x2)
        w1, w2 = (self.w1, self.w2)
        w1_ = 2.0 * np.ones(len(x1))
        w2_ = 2.0 * np.ones(len(x2))
        d1 = DescrStatsW(x1)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1_, w2_))[:2], stats.ttest_ind(np.r_[x1, x1], np.r_[x2, x2]))

    def test_weightstats_2(self):
        x1, x2 = (self.x1, self.x2)
        w1, w2 = (self.w1, self.w2)
        d1 = DescrStatsW(x1)
        d1w = DescrStatsW(x1, weights=w1)
        d2w = DescrStatsW(x2, weights=w2)
        x1r = d1w.asrepeats()
        x2r = d2w.asrepeats()
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2], stats.ttest_ind(x1r, x2r), 14)
        assert_almost_equal(x2r.mean(0), d2w.mean, 14)
        assert_almost_equal(x2r.var(), d2w.var, 14)
        assert_almost_equal(x2r.std(), d2w.std, 14)
        assert_almost_equal(np.cov(x2r, bias=1), d2w.cov, 14)
        assert_almost_equal(d1.ttest_mean(3)[:2], stats.ttest_1samp(x1, 3), 11)
        assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)

    def test_weightstats_3(self):
        x1_2d, x2_2d = (self.x1_2d, self.x2_2d)
        w1, w2 = (self.w1, self.w2)
        d1w_2d = DescrStatsW(x1_2d, weights=w1)
        d2w_2d = DescrStatsW(x2_2d, weights=w2)
        x1r_2d = d1w_2d.asrepeats()
        x2r_2d = d2w_2d.asrepeats()
        assert_almost_equal(x2r_2d.mean(0), d2w_2d.mean, 14)
        assert_almost_equal(x2r_2d.var(0), d2w_2d.var, 14)
        assert_almost_equal(x2r_2d.std(0), d2w_2d.std, 14)
        assert_almost_equal(np.cov(x2r_2d.T, bias=1), d2w_2d.cov, 14)
        assert_almost_equal(np.corrcoef(x2r_2d.T), d2w_2d.corrcoef, 14)
        t, p, d = d1w_2d.ttest_mean(3)
        assert_almost_equal([t, p], stats.ttest_1samp(x1r_2d, 3), 11)
        cm = CompareMeans(d1w_2d, d2w_2d)
        ressm = cm.ttest_ind()
        resss = stats.ttest_ind(x1r_2d, x2r_2d)
        assert_almost_equal(ressm[:2], resss, 14)

    def test_weightstats_ddof_tests(self):
        x1_2d = self.x1_2d
        w1 = self.w1
        d1w_d0 = DescrStatsW(x1_2d, weights=w1, ddof=0)
        d1w_d1 = DescrStatsW(x1_2d, weights=w1, ddof=1)
        d1w_d2 = DescrStatsW(x1_2d, weights=w1, ddof=2)
        res0 = d1w_d0.ttest_mean()
        res1 = d1w_d1.ttest_mean()
        res2 = d1w_d2.ttest_mean()
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)
        res0 = d1w_d0.ttest_mean(0.5)
        res1 = d1w_d1.ttest_mean(0.5)
        res2 = d1w_d2.ttest_mean(0.5)
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)
        res0 = d1w_d0.tconfint_mean()
        res1 = d1w_d1.tconfint_mean()
        res2 = d1w_d2.tconfint_mean()
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_comparemeans_convenient_interface(self):
        x1_2d, x2_2d = (self.x1_2d, self.x2_2d)
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))

    def test_comparemeans_convenient_interface_1d(self):
        x1_2d, x2_2d = (self.x1, self.x2)
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))