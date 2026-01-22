import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class CheckWeightstats1dMixin:

    def test_basic(self):
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(x1r.mean(0), d1w.mean, 14)
        assert_almost_equal(x1r.var(0, ddof=d1w.ddof), d1w.var, 14)
        assert_almost_equal(x1r.std(0, ddof=d1w.ddof), d1w.std, 14)
        var1 = d1w.var_ddof(ddof=1)
        assert_almost_equal(x1r.var(0, ddof=1), var1, 14)
        std1 = d1w.std_ddof(ddof=1)
        assert_almost_equal(x1r.std(0, ddof=1), std1, 14)
        assert_almost_equal(np.cov(x1r.T, bias=1 - d1w.ddof), d1w.cov, 14)

    def test_ttest(self):
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)

    def test_ttest_2sample(self):
        x1, x2 = (self.x1, self.x2)
        x1r, x2r = (self.x1r, self.x2r)
        w1, w2 = (self.w1, self.w2)
        res_sp = stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2], res_sp, 14)
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=1))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1), DescrStatsW(x2, weights=w2, ddof=2))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)
        cm0 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=0))
        cm1 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=1))
        cm2 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1), DescrStatsW(x2, weights=w2, ddof=2))
        res0 = cm0.ttest_ind(usevar='unequal')
        res1 = cm1.ttest_ind(usevar='unequal')
        res2 = cm2.ttest_ind(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)
        res0 = cm0.tconfint_diff(usevar='pooled')
        res1 = cm1.tconfint_diff(usevar='pooled')
        res2 = cm2.tconfint_diff(usevar='pooled')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)
        res0 = cm0.tconfint_diff(usevar='unequal')
        res1 = cm1.tconfint_diff(usevar='unequal')
        res2 = cm2.tconfint_diff(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_confint_mean(self):
        d1w = self.d1w
        alpha = 0.05
        low, upp = d1w.tconfint_mean()
        t, p, d = d1w.ttest_mean(low)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(upp)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(np.vstack((low, upp)))
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)