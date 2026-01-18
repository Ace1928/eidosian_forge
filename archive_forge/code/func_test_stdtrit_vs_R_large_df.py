import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.special import stdtr, stdtrit, ndtr, ndtri
def test_stdtrit_vs_R_large_df():
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    p = 0.1
    res = stdtrit(df, p)
    res_R = [-1.2815515656292593, -1.2815515655454472, -1.2815515655446008, -1.2815515655446008]
    assert_allclose(res, res_R, rtol=1e-15)
    assert_equal(res[3], ndtri(0.1))