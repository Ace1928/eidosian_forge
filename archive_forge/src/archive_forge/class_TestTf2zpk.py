import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestTf2zpk:

    @pytest.mark.parametrize('dt', (np.float64, np.complex128))
    def test_simple(self, dt):
        z_r = np.array([0.5, -0.5])
        p_r = np.array([1j / np.sqrt(2), -1j / np.sqrt(2)])
        z_r.sort()
        p_r.sort()
        b = np.poly(z_r).astype(dt)
        a = np.poly(p_r).astype(dt)
        z, p, k = tf2zpk(b, a)
        z.sort()
        p = p[np.argsort(p.imag)]
        assert_array_almost_equal(z, z_r)
        assert_array_almost_equal(p, p_r)
        assert_array_almost_equal(k, 1.0)
        assert k.dtype == dt

    def test_bad_filter(self):
        with suppress_warnings():
            warnings.simplefilter('error', BadCoefficients)
            assert_raises(BadCoefficients, tf2zpk, [1e-15], [1.0, 1.0])