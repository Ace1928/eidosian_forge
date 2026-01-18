import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.special._testutils import FuncData
from scipy.special import gamma, gammaln, loggamma
def test_real_dispatch():
    x = np.logspace(-10, 10) + 0.5
    dataset = np.vstack((x, gammaln(x))).T
    FuncData(loggamma, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
    assert_(loggamma(0) == np.inf)
    assert_(np.isnan(loggamma(-1)))