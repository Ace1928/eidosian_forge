import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
class SphericalDerivativesTestCase:

    def fundamental_theorem(self, n, a, b):
        integral, tolerance = quad(lambda z: self.df(n, z), a, b)
        assert_allclose(integral, self.f(n, b) - self.f(n, a), atol=tolerance)

    @pytest.mark.slow
    def test_fundamental_theorem_0(self):
        self.fundamental_theorem(0, 3.0, 15.0)

    @pytest.mark.slow
    def test_fundamental_theorem_7(self):
        self.fundamental_theorem(7, 0.5, 1.2)