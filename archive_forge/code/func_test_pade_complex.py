from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from scipy.interpolate import pade
def test_pade_complex():
    x = 0.2 + 0.6j
    an = [1.0, x, -x * x.conjugate(), x.conjugate() * x ** 2 + x * x.conjugate() ** 2, -x ** 3 * x.conjugate() - 3 * (x * x.conjugate()) ** 2 - x * x.conjugate() ** 3]
    nump, denomp = pade(an, 1, 1)
    assert_array_almost_equal(nump.c, [x + x.conjugate(), 1.0])
    assert_array_almost_equal(denomp.c, [x.conjugate(), 1.0])
    nump, denomp = pade(an, 1, 2)
    assert_array_almost_equal(nump.c, [x ** 2, 2 * x + x.conjugate(), 1.0])
    assert_array_almost_equal(denomp.c, [x + x.conjugate(), 1.0])
    nump, denomp = pade(an, 2, 2)
    assert_array_almost_equal(nump.c, [x ** 2 + x * x.conjugate() + x.conjugate() ** 2, 2 * (x + x.conjugate()), 1.0])
    assert_array_almost_equal(denomp.c, [x.conjugate() ** 2, x + 2 * x.conjugate(), 1.0])