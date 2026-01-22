import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class TestPartialFractionExpansion:

    @staticmethod
    def assert_rp_almost_equal(r, p, r_true, p_true, decimal=7):
        r_true = np.asarray(r_true)
        p_true = np.asarray(p_true)
        distance = np.hypot(abs(p[:, None] - p_true), abs(r[:, None] - r_true))
        rows, cols = linear_sum_assignment(distance)
        assert_almost_equal(p[rows], p_true[cols], decimal=decimal)
        assert_almost_equal(r[rows], r_true[cols], decimal=decimal)

    def test_compute_factors(self):
        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1])
        assert_equal(len(factors), 3)
        assert_almost_equal(factors[0], np.poly([2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[2], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))
        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1], include_powers=True)
        assert_equal(len(factors), 6)
        assert_almost_equal(factors[0], np.poly([1, 1, 2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 2, 2, 3]))
        assert_almost_equal(factors[2], np.poly([2, 2, 3]))
        assert_almost_equal(factors[3], np.poly([1, 1, 1, 2, 3]))
        assert_almost_equal(factors[4], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[5], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))

    def test_group_poles(self):
        unique, multiplicity = _group_poles([1.0, 1.001, 1.003, 2.0, 2.003, 3.0], 0.1, 'min')
        assert_equal(unique, [1.0, 2.0, 3.0])
        assert_equal(multiplicity, [3, 2, 1])

    def test_residue_general(self):
        r, p, k = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        assert_almost_equal(r, [1.332, -0.6653, -1.4167], decimal=4)
        assert_almost_equal(p, [-0.4093, -1.1644, 1.5737], decimal=4)
        assert_almost_equal(k, [-1.25], decimal=4)
        r, p, k = residue([-4, 8], [1, 6, 8])
        assert_almost_equal(r, [8, -12])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)
        r, p, k = residue([4, 1], [1, -1, -2])
        assert_almost_equal(r, [1, 3])
        assert_almost_equal(p, [-1, 2])
        assert_equal(k.size, 0)
        r, p, k = residue([4, 3], [2, -3.4, 1.98, -0.406])
        self.assert_rp_almost_equal(r, p, [-18.125 - 13.125j, -18.125 + 13.125j, 36.25], [0.5 - 0.2j, 0.5 + 0.2j, 0.7])
        assert_equal(k.size, 0)
        r, p, k = residue([2, 1], [1, 5, 8, 4])
        self.assert_rp_almost_equal(r, p, [-1, 1, 3], [-1, -2, -2])
        assert_equal(k.size, 0)
        r, p, k = residue([3, -1.1, 0.88, -2.396, 1.348], [1, -0.7, -0.14, 0.048])
        assert_almost_equal(r, [-3, 4, 1])
        assert_almost_equal(p, [0.2, -0.3, 0.8])
        assert_almost_equal(k, [3, 1])
        r, p, k = residue([1], [1, 2, -3])
        assert_almost_equal(r, [0.25, -0.25])
        assert_almost_equal(p, [1, -3])
        assert_equal(k.size, 0)
        r, p, k = residue([1, 0, -5], [1, 0, 0, 0, -1])
        self.assert_rp_almost_equal(r, p, [1, 1.5j, -1.5j, -1], [-1, -1j, 1j, 1])
        assert_equal(k.size, 0)
        r, p, k = residue([3, 8, 6], [1, 3, 3, 1])
        self.assert_rp_almost_equal(r, p, [1, 2, 3], [-1, -1, -1])
        assert_equal(k.size, 0)
        r, p, k = residue([3, -1], [1, -3, 2])
        assert_almost_equal(r, [-2, 5])
        assert_almost_equal(p, [1, 2])
        assert_equal(k.size, 0)
        r, p, k = residue([2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-4, 13])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [2])
        r, p, k = residue([7, 2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-11, 69])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [7, 23])
        r, p, k = residue([2, 3, -1], [1, -3, 4, -2])
        self.assert_rp_almost_equal(r, p, [4, -1 + 3.5j, -1 - 3.5j], [1, 1 - 1j, 1 + 1j])
        assert_almost_equal(k.size, 0)

    def test_residue_leading_zeros(self):
        r0, p0, k0 = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residue([0, 5, 3, -2, 7], [-4, 0, 8, 3])
        r2, p2, k2 = residue([5, 3, -2, 7], [0, -4, 0, 8, 3])
        r3, p3, k3 = residue([0, 0, 5, 3, -2, 7], [0, 0, 0, -4, 0, 8, 3])
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_resiude_degenerate(self):
        r, p, k = residue([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)
        r, p, k = residue(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)
        with pytest.raises(ValueError, match='Denominator `a` is zero.'):
            residue(1, 0)

    def test_residuez_general(self):
        r, p, k = residuez([1, 6, 6, 2], [1, -(2 + 1j), 1 + 2j, -1j])
        self.assert_rp_almost_equal(r, p, [-2 + 2.5j, 7.5 + 7.5j, -4.5 - 12j], [1j, 1, 1])
        assert_almost_equal(k, [2j])
        r, p, k = residuez([1, 2, 1], [1, -1, 0.3561])
        self.assert_rp_almost_equal(r, p, [-0.9041 - 5.9928j, -0.9041 + 5.9928j], [0.5 + 0.3257j, 0.5 - 0.3257j], decimal=4)
        assert_almost_equal(k, [2.8082], decimal=4)
        r, p, k = residuez([1, -1], [1, -5, 6])
        assert_almost_equal(r, [-1, 2])
        assert_almost_equal(p, [2, 3])
        assert_equal(k.size, 0)
        r, p, k = residuez([2, 3, 4], [1, 3, 3, 1])
        self.assert_rp_almost_equal(r, p, [4, -5, 3], [-1, -1, -1])
        assert_equal(k.size, 0)
        r, p, k = residuez([1, -10, -4, 4], [2, -2, -4])
        assert_almost_equal(r, [0.5, -1.5])
        assert_almost_equal(p, [-1, 2])
        assert_almost_equal(k, [1.5, -1])
        r, p, k = residuez([18], [18, 3, -4, -1])
        self.assert_rp_almost_equal(r, p, [0.36, 0.24, 0.4], [0.5, -1 / 3, -1 / 3])
        assert_equal(k.size, 0)
        r, p, k = residuez([2, 3], np.polymul([1, -1 / 2], [1, 1 / 4]))
        assert_almost_equal(r, [-10 / 3, 16 / 3])
        assert_almost_equal(p, [-0.25, 0.5])
        assert_equal(k.size, 0)
        r, p, k = residuez([1, -2, 1], [1, -1])
        assert_almost_equal(r, [0])
        assert_almost_equal(p, [1])
        assert_almost_equal(k, [1, -1])
        r, p, k = residuez(1, [1, -1j])
        assert_almost_equal(r, [1])
        assert_almost_equal(p, [1j])
        assert_equal(k.size, 0)
        r, p, k = residuez(1, [1, -1, 0.25])
        assert_almost_equal(r, [0, 1])
        assert_almost_equal(p, [0.5, 0.5])
        assert_equal(k.size, 0)
        r, p, k = residuez(1, [1, -0.75, 0.125])
        assert_almost_equal(r, [-1, 2])
        assert_almost_equal(p, [0.25, 0.5])
        assert_equal(k.size, 0)
        r, p, k = residuez([1, 6, 2], [1, -2, 1])
        assert_almost_equal(r, [-10, 9])
        assert_almost_equal(p, [1, 1])
        assert_almost_equal(k, [2])
        r, p, k = residuez([6, 2], [1, -2, 1])
        assert_almost_equal(r, [-2, 8])
        assert_almost_equal(p, [1, 1])
        assert_equal(k.size, 0)
        r, p, k = residuez([1, 6, 6, 2], [1, -2, 1])
        assert_almost_equal(r, [-24, 15])
        assert_almost_equal(p, [1, 1])
        assert_almost_equal(k, [10, 2])
        r, p, k = residuez([1, 0, 1], [1, 0, 0, 0, 0, -1])
        self.assert_rp_almost_equal(r, p, [0.2618 + 0.1902j, 0.2618 - 0.1902j, 0.4, 0.0382 - 0.1176j, 0.0382 + 0.1176j], [-0.809 + 0.5878j, -0.809 - 0.5878j, 1.0, 0.309 + 0.9511j, 0.309 - 0.9511j], decimal=4)
        assert_equal(k.size, 0)

    def test_residuez_trailing_zeros(self):
        r0, p0, k0 = residuez([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residuez([5, 3, -2, 7, 0], [-4, 0, 8, 3])
        r2, p2, k2 = residuez([5, 3, -2, 7], [-4, 0, 8, 3, 0])
        r3, p3, k3 = residuez([5, 3, -2, 7, 0, 0], [-4, 0, 8, 3, 0, 0, 0])
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_residuez_degenerate(self):
        r, p, k = residuez([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)
        r, p, k = residuez(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)
        with pytest.raises(ValueError, match='Denominator `a` is zero.'):
            residuez(1, 0)
        with pytest.raises(ValueError, match='First coefficient of determinant `a` must be non-zero.'):
            residuez(1, [0, 1, 2, 3])

    def test_inverse_unique_roots_different_rtypes(self):
        r = [3 / 10, -1 / 6, -2 / 15]
        p = [0, -2, -5]
        k = []
        b_expected = [0, 1, 3]
        a_expected = [1, 7, 10, 0]
        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)
            assert_allclose(a, a_expected)
            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)
            assert_allclose(a, a_expected)

    def test_inverse_repeated_roots_different_rtypes(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]
        p = [0, -2, -2, -5]
        k = []
        b_expected = [0, 0, 1, 3]
        b_expected_z = [-1 / 6, -2 / 3, 11 / 6, 3]
        a_expected = [1, 9, 24, 20, 0]
        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected, atol=1e-14)
            assert_allclose(a, a_expected)
            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected_z, atol=1e-14)
            assert_allclose(a, a_expected)

    def test_inverse_bad_rtype(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]
        p = [0, -2, -2, -5]
        k = []
        with pytest.raises(ValueError, match='`rtype` must be one of'):
            invres(r, p, k, rtype='median')
        with pytest.raises(ValueError, match='`rtype` must be one of'):
            invresz(r, p, k, rtype='median')

    def test_invresz_one_coefficient_bug(self):
        r = [1]
        p = [2]
        k = [0]
        b, a = invresz(r, p, k)
        assert_allclose(b, [1.0])
        assert_allclose(a, [1.0, -2.0])

    def test_invres(self):
        b, a = invres([1], [1], [])
        assert_almost_equal(b, [1])
        assert_almost_equal(a, [1, -1])
        b, a = invres([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])
        b, a = invres([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
        assert_almost_equal(b, [1, -1 - 1j, 1 - 2j, 0.5 - 3j, 10])
        assert_almost_equal(a, [1, -3 - 1j, 4])
        b, a = invres([-1, 2, 1j, 3 - 1j, 4, -2], [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
        assert_almost_equal(b, [4 - 1j, -28 + 16j, 40 - 62j, 100 + 24j, -292 + 219j, 192 - 268j])
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j, 108 - 54j, -81 + 108j])
        b, a = invres([-1, 1j], [1, 1], [1, 2])
        assert_almost_equal(b, [1, 0, -4, 3 + 1j])
        assert_almost_equal(a, [1, -2, 1])

    def test_invresz(self):
        b, a = invresz([1], [1], [])
        assert_almost_equal(b, [1])
        assert_almost_equal(a, [1, -1])
        b, a = invresz([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])
        b, a = invresz([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
        assert_almost_equal(b, [2.5, -3 - 1j, 1 - 2j, -1 - 3j, 12])
        assert_almost_equal(a, [1, -3 - 1j, 4])
        b, a = invresz([-1, 2, 1j, 3 - 1j, 4, -2], [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
        assert_almost_equal(b, [6, -50 + 11j, 100 - 72j, 80 + 58j, -354 + 228j, 234 - 297j])
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j, 108 - 54j, -81 + 108j])
        b, a = invresz([-1, 1j], [1, 1], [1, 2])
        assert_almost_equal(b, [1j, 1, -3, 2])
        assert_almost_equal(a, [1, -2, 1])

    def test_inverse_scalar_arguments(self):
        b, a = invres(1, 1, 1)
        assert_almost_equal(b, [1, 0])
        assert_almost_equal(a, [1, -1])
        b, a = invresz(1, 1, 1)
        assert_almost_equal(b, [2, -1])
        assert_almost_equal(a, [1, -1])