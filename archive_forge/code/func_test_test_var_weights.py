import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_test_var_weights(self):
    assert_almost_equal(self.res1.test_var(3, return_weights=1)[2], self.res2.test_var_weights, 4)