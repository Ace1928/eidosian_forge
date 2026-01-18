import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_kn_recurrence_real(self):
    n = np.array([1, 2, 3, 7, 12])
    x = 0.12
    assert_allclose((-1) ** (n - 1) * spherical_kn(n - 1, x) - (-1) ** (n + 1) * spherical_kn(n + 1, x), (-1) ** n * (2 * n + 1) / x * spherical_kn(n, x))