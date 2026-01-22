import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestSim2(CheckExternalMixin):
    mean = [-0.2170406, -0.2387543]
    sum = [-6.8383999, -7.5225444]
    var = [1.77426344, 0.61933542]
    std = [1.3320148, 0.78697867]
    quantiles = np.column_stack((np.r_[-2.55277, -1.40479, -0.6104, 0.5274, 2.66246], np.r_[-1.49263, -1.15403, -0.16231, 0.16464, 1.83062]))

    @classmethod
    def setup_class(cls):
        np.random.seed(2249)
        cls.data = np.random.normal(size=(20, 2))
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()