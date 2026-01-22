import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
class TestSphericalIn:

    def test_spherical_in_exact(self):
        x = np.array([0.12, 1.23, 12.34, 123.45])
        assert_allclose(spherical_in(2, x), (1 / x + 3 / x ** 3) * sinh(x) - 3 / x ** 2 * cosh(x))

    def test_spherical_in_recurrence_real(self):
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1, x), (2 * n + 1) / x * spherical_in(n, x))

    def test_spherical_in_recurrence_complex(self):
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1, x), (2 * n + 1) / x * spherical_in(n, x))

    def test_spherical_in_inf_real(self):
        n = 5
        x = np.array([-inf, inf])
        assert_allclose(spherical_in(n, x), np.array([-inf, inf]))

    def test_spherical_in_inf_complex(self):
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf * (1 + 1j)])
        assert_allclose(spherical_in(n, x), np.array([-inf, inf, nan]))

    def test_spherical_in_at_zero(self):
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_in(n, x), np.array([1, 0, 0, 0, 0, 0]))