import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, \
def test_x_dims(self):
    np.random.seed(3975)
    x_n = stats.norm.rvs(size=500)
    data = x_n
    d_ks_norm, p_norm = lilliefors(data, dist='norm', pvalmethod='approx')
    assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
    assert_almost_equal(p_norm, 0.64175, decimal=3)
    data = x_n.reshape(-1, 1)
    d_ks_norm, p_norm = lilliefors(data, dist='norm', pvalmethod='approx')
    assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
    assert_almost_equal(p_norm, 0.64175, decimal=3)
    data = np.array([x_n, x_n]).T
    with pytest.raises(ValueError):
        lilliefors(data, dist='norm', pvalmethod='approx')
    data = pd.DataFrame(data=x_n)
    d_ks_norm, p_norm = lilliefors(data, dist='norm', pvalmethod='approx')
    assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
    assert_almost_equal(p_norm, 0.64175, decimal=3)
    data = pd.DataFrame(data=[x_n, x_n])
    with pytest.raises(ValueError):
        lilliefors(data, dist='norm', pvalmethod='approx')
    data = pd.DataFrame(data=x_n.reshape(-1, 1).T)
    with pytest.raises(ValueError):
        lilliefors(data, dist='norm', pvalmethod='approx')