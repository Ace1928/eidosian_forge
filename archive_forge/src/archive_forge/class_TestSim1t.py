import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestSim1t(CheckExternalMixin):
    mean = 5.05103296
    sum = 156.573464
    var = 9.9711934
    std = 3.15771965
    quantiles = np.r_[0, 1, 5, 8, 9]

    @classmethod
    def setup_class(cls):
        np.random.seed(9876789)
        cls.data = np.random.randint(0, 10, size=20)
        cls.data[15:20] = cls.data[0:5]
        cls.data[18:20] = cls.data[15:17]
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()