import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_sph_jn(self):
    s1 = np.empty((2, 3))
    x = 0.2
    s1[0][0] = spherical_jn(0, x)
    s1[0][1] = spherical_jn(1, x)
    s1[0][2] = spherical_jn(2, x)
    s1[1][0] = spherical_jn(0, x, derivative=True)
    s1[1][1] = spherical_jn(1, x, derivative=True)
    s1[1][2] = spherical_jn(2, x, derivative=True)
    s10 = -s1[0][1]
    s11 = s1[0][0] - 2.0 / 0.2 * s1[0][1]
    s12 = s1[0][1] - 3.0 / 0.2 * s1[0][2]
    assert_array_almost_equal(s1[0], [0.9933466539753061, 0.06640038067032224, 0.0026590560795273855], 12)
    assert_array_almost_equal(s1[1], [s10, s11, s12], 12)