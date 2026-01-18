import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_weightstats_1(self):
    x1, x2 = (self.x1, self.x2)
    w1, w2 = (self.w1, self.w2)
    w1_ = 2.0 * np.ones(len(x1))
    w2_ = 2.0 * np.ones(len(x2))
    d1 = DescrStatsW(x1)
    assert_almost_equal(ttest_ind(x1, x2, weights=(w1_, w2_))[:2], stats.ttest_ind(np.r_[x1, x1], np.r_[x2, x2]))