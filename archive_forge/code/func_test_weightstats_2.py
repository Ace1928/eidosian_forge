import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
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