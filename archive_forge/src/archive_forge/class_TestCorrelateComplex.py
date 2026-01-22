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
@pytest.mark.parametrize('dt', [np.csingle, np.cdouble, np.clongdouble])
class TestCorrelateComplex:

    def decimal(self, dt):
        if dt == np.clongdouble:
            dt = np.cdouble
        return int(2 * np.finfo(dt).precision / 3)

    def _setup_rank1(self, dt, mode):
        np.random.seed(9)
        a = np.random.randn(10).astype(dt)
        a += 1j * np.random.randn(10).astype(dt)
        b = np.random.randn(8).astype(dt)
        b += 1j * np.random.randn(8).astype(dt)
        y_r = (correlate(a.real, b.real, mode=mode) + correlate(a.imag, b.imag, mode=mode)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag, mode=mode) + correlate(a.imag, b.real, mode=mode))
        return (a, b, y_r)

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'valid')
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[::-1].conj(), decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'same')
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'full')
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_swap_full(self, dt):
        d = np.array([0.0 + 0j, 1.0 + 1j, 2.0 + 2j], dtype=dt)
        k = np.array([1.0 + 3j, 2.0 + 4j, 3.0 + 5j, 4.0 + 6j], dtype=dt)
        y = correlate(d, k)
        assert_equal(y, [0.0 + 0j, 10.0 - 2j, 28.0 - 6j, 22.0 - 6j, 16.0 - 6j, 8.0 - 4j])

    def test_swap_same(self, dt):
        d = [0.0 + 0j, 1.0 + 1j, 2.0 + 2j]
        k = [1.0 + 3j, 2.0 + 4j, 3.0 + 5j, 4.0 + 6j]
        y = correlate(d, k, mode='same')
        assert_equal(y, [10.0 - 2j, 28.0 - 6j, 22.0 - 6j])

    def test_rank3(self, dt):
        a = np.random.randn(10, 8, 6).astype(dt)
        a += 1j * np.random.randn(10, 8, 6).astype(dt)
        b = np.random.randn(8, 6, 4).astype(dt)
        b += 1j * np.random.randn(8, 6, 4).astype(dt)
        y_r = (correlate(a.real, b.real) + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)

    def test_rank0(self, dt):
        a = np.array(np.random.randn()).astype(dt)
        a += 1j * np.array(np.random.randn()).astype(dt)
        b = np.array(np.random.randn()).astype(dt)
        b += 1j * np.array(np.random.randn()).astype(dt)
        y_r = (correlate(a.real, b.real) + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * np.array(-correlate(a.real, b.imag) + correlate(a.imag, b.real))
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)
        assert_equal(correlate([1], [2j]), correlate(1, 2j))
        assert_equal(correlate([2j], [3j]), correlate(2j, 3j))
        assert_equal(correlate([3j], [4]), correlate(3j, 4))