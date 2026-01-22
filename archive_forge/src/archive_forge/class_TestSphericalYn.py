import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
class TestSphericalYn:

    def test_spherical_yn_exact(self):
        x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
        assert_allclose(spherical_yn(2, x), (1 / x - 3 / x ** 3) * cos(x) - 3 / x ** 2 * sin(x))

    def test_spherical_yn_recurrence_real(self):
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x), (2 * n + 1) / x * spherical_yn(n, x))

    def test_spherical_yn_recurrence_complex(self):
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x), (2 * n + 1) / x * spherical_yn(n, x))

    def test_spherical_yn_inf_real(self):
        n = 6
        x = np.array([-inf, inf])
        assert_allclose(spherical_yn(n, x), np.array([0, 0]))

    def test_spherical_yn_inf_complex(self):
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf * (1 + 1j)])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
            assert_allclose(spherical_yn(n, x), np.array([0, 0, inf * (1 + 1j)]))

    def test_spherical_yn_at_zero(self):
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_yn(n, x), np.full(n.shape, -inf))

    def test_spherical_yn_at_zero_complex(self):
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0 + 0j
        assert_allclose(spherical_yn(n, x), np.full(n.shape, nan))