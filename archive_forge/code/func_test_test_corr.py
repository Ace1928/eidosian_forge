import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_test_corr(self):
    assert_almost_equal(self.mvres1.test_corr(0.5), self.res2.test_corr, 4)