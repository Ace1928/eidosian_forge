import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
class TestSphericalKnDerivatives(SphericalDerivativesTestCase):

    def f(self, n, z):
        return spherical_kn(n, z)

    def df(self, n, z):
        return spherical_kn(n, z, derivative=True)