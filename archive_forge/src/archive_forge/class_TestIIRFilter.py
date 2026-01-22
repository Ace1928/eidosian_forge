import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestIIRFilter:

    def test_symmetry(self):
        for N in np.arange(1, 26):
            for ftype in ('butter', 'bessel', 'cheby1', 'cheby2', 'ellip'):
                z, p, k = iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='zpk')
                assert_array_equal(sorted(z), sorted(z.conj()))
                assert_array_equal(sorted(p), sorted(p.conj()))
                assert_equal(k, np.real(k))
                b, a = iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='ba')
                assert_(issubclass(b.dtype.type, np.floating))
                assert_(issubclass(a.dtype.type, np.floating))

    def test_int_inputs(self):
        k = iirfilter(24, 100, btype='low', analog=True, ftype='bessel', output='zpk')[2]
        k2 = 9.999999999999989e+47
        assert_allclose(k, k2)

    def test_invalid_wn_size(self):
        assert_raises(ValueError, iirfilter, 1, [0.1, 0.9], btype='low')
        assert_raises(ValueError, iirfilter, 1, [0.2, 0.5], btype='high')
        assert_raises(ValueError, iirfilter, 1, 0.2, btype='bp')
        assert_raises(ValueError, iirfilter, 1, 400, btype='bs', analog=True)

    def test_invalid_wn_range(self):
        assert_raises(ValueError, iirfilter, 1, 2, btype='low')
        assert_raises(ValueError, iirfilter, 1, [0.5, 1], btype='band')
        assert_raises(ValueError, iirfilter, 1, [0.0, 0.5], btype='band')
        assert_raises(ValueError, iirfilter, 1, -1, btype='high')
        assert_raises(ValueError, iirfilter, 1, [1, 2], btype='band')
        assert_raises(ValueError, iirfilter, 1, [10, 20], btype='stop')
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, 0, btype='low', analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, -1, btype='low', analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, [0, 100], analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, [-1, 100], analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, [10, 0], analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirfilter(2, [10, -1], analog=True)

    def test_analog_sos(self):
        sos = [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]
        sos2 = iirfilter(N=1, Wn=1, btype='low', analog=True, output='sos')
        assert_array_almost_equal(sos, sos2)

    def test_wn1_ge_wn0(self):
        with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
            iirfilter(2, [0.5, 0.5])
        with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
            iirfilter(2, [0.6, 0.5])