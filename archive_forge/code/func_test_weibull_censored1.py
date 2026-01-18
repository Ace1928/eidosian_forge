import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_weibull_censored1():
    s = '3,5,6*,8,10*,11*,15,20*,22,23,27*,29,32,35,40,26,28,33*,21,24*'
    times, cens = zip(*[(float(t[0]), len(t) == 2) for t in [w.split('*') for w in s.split(',')]])
    data = CensoredData.right_censored(times, cens)
    c, loc, scale = weibull_min.fit(data, floc=0)
    assert_allclose(c, 2.149, rtol=0.001)
    assert loc == 0
    assert_allclose(scale, 28.99, rtol=0.001)
    data2 = CensoredData.left_censored(-np.array(times), cens)
    c2, loc2, scale2 = weibull_max.fit(data2, floc=0)
    assert_allclose(c2, 2.149, rtol=0.001)
    assert loc2 == 0
    assert_allclose(scale2, 28.99, rtol=0.001)