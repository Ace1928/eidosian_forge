import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_sph_kn(self):
    kn = np.empty((2, 3))
    x = 0.2
    kn[0][0] = spherical_kn(0, x)
    kn[0][1] = spherical_kn(1, x)
    kn[0][2] = spherical_kn(2, x)
    kn[1][0] = spherical_kn(0, x, derivative=True)
    kn[1][1] = spherical_kn(1, x, derivative=True)
    kn[1][2] = spherical_kn(2, x, derivative=True)
    kn0 = -kn[0][1]
    kn1 = -kn[0][0] - 2.0 / 0.2 * kn[0][1]
    kn2 = -kn[0][1] - 3.0 / 0.2 * kn[0][2]
    assert_array_almost_equal(kn[0], [6.430296297844567, 38.5817777870674, 585.1569631038556], 12)
    assert_array_almost_equal(kn[1], [kn0, kn1, kn2], 9)