import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_pval_bounds(self):
    x = stats.norm.ppf((np.arange(10.0) + 0.5) / 10)
    d_ks_n, p_n = lilliefors(x, dist='norm', pvalmethod='approx')
    x = stats.expon.ppf((np.arange(10.0) + 0.5) / 10)
    d_ks_e, p_e = lilliefors(x, dist='exp', pvalmethod='approx')
    assert_almost_equal(p_n, 0.99, decimal=7)
    assert_almost_equal(p_e, 0.99, decimal=7)