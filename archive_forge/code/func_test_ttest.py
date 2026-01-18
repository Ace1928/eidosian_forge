import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_ttest(self):
    x1r = self.x1r
    d1w = self.d1w
    assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)