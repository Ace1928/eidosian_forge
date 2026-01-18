import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_in_d_zero(self):
    n = np.array([1, 2, 3, 7, 15])
    assert_allclose(spherical_in(n, 0, derivative=True), np.zeros(5))