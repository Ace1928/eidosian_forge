import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
def test_squared_normal_chi2():
    cdftr = squarenormalg.cdf(xx, loc=l, scale=s)
    sfctr = 1 - squarenormalg.sf(xx, loc=l, scale=s)
    cdfst = stats.chi2.cdf(xx, 1)
    assert_almost_equal(cdfst, cdftr, 14)
    assert_almost_equal(cdfst, sfctr, 14)