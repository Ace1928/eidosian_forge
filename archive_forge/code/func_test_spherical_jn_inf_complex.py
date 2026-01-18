import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
def test_spherical_jn_inf_complex(self):
    n = 7
    x = np.array([-inf + 0j, inf + 0j, inf * (1 + 1j)])
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
        assert_allclose(spherical_jn(n, x), np.array([0, 0, inf * (1 + 1j)]))