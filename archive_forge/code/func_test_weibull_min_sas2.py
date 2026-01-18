import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_weibull_min_sas2():
    days = np.array([143, 164, 188, 188, 190, 192, 206, 209, 213, 216, 220, 227, 230, 234, 246, 265, 304, 216, 244])
    data = CensoredData.right_censored(days, [0] * (len(days) - 2) + [1] * 2)
    c, loc, scale = weibull_min.fit(data, 1, loc=100, scale=100, optimizer=optimizer)
    assert_allclose(c, 2.7112, rtol=0.0005)
    assert_allclose(loc, 122.03, rtol=0.0005)
    assert_allclose(scale, 108.37, rtol=0.0005)