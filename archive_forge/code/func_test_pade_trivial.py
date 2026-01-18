from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from scipy.interpolate import pade
def test_pade_trivial():
    nump, denomp = pade([1.0], 0)
    assert_array_equal(nump.c, [1.0])
    assert_array_equal(denomp.c, [1.0])
    nump, denomp = pade([1.0], 0, 0)
    assert_array_equal(nump.c, [1.0])
    assert_array_equal(denomp.c, [1.0])