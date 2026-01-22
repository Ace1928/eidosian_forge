import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestWeightstats2d_nobs(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        n1, n2 = (20, 30)
        m1, m2 = (1, 1.2)
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        cls.x1, cls.x2 = (x1, x2)
        cls.w1, cls.w2 = (w1, w2)
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=0)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()