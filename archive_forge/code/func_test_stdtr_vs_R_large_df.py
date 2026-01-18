import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.special import stdtr, stdtrit, ndtr, ndtri
def test_stdtr_vs_R_large_df():
    df = [10000000000.0, 1000000000000.0, 1e+120, np.inf]
    t = 1.0
    res = stdtr(df, t)
    res_R = [0.8413447460564446, 0.8413447460684218, 0.8413447460685428, 0.8413447460685429]
    assert_allclose(res, res_R, rtol=2e-15)
    assert_equal(res[3], ndtr(1.0))