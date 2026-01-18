import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
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