from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from scipy.interpolate import pade
def test_pade_ints():
    an_int = [1, 2, 3, 4]
    an_flt = [1.0, 2.0, 3.0, 4.0]
    for i in range(0, len(an_int)):
        for j in range(0, len(an_int) - i):
            nump_int, denomp_int = pade(an_int, i, j)
            nump_flt, denomp_flt = pade(an_flt, i, j)
            assert_array_equal(nump_int.c, nump_flt.c)
            assert_array_equal(denomp_int.c, denomp_flt.c)