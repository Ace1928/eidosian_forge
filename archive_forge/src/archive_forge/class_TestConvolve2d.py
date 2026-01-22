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
class TestConvolve2d(_TestConvolve2d):

    def test_same_mode(self):
        e = [[1, 2, 3], [3, 4, 5]]
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        g = convolve2d(e, f, 'same')
        h = array([[22, 28, 34], [80, 98, 116]])
        assert_array_equal(g, h)

    def test_valid_mode2(self):
        e = [[1, 2, 3], [3, 4, 5]]
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        expected = [[62, 80, 98, 116, 134]]
        out = convolve2d(e, f, 'valid')
        assert_array_equal(out, expected)
        out = convolve2d(f, e, 'valid')
        assert_array_equal(out, expected)
        e = [[1 + 1j, 2 - 3j], [3 + 1j, 4 + 0j]]
        f = [[2 - 1j, 3 + 2j, 4 + 0j], [4 - 0j, 5 + 1j, 6 - 3j]]
        expected = [[27 - 1j, 46.0 + 2j]]
        out = convolve2d(e, f, 'valid')
        assert_array_equal(out, expected)
        out = convolve2d(f, e, 'valid')
        assert_array_equal(out, expected)

    def test_consistency_convolve_funcs(self):
        a = np.arange(5)
        b = np.array([3.2, 1.4, 3])
        for mode in ['full', 'valid', 'same']:
            assert_almost_equal(np.convolve(a, b, mode=mode), signal.convolve(a, b, mode=mode))
            assert_almost_equal(np.squeeze(signal.convolve2d([a], [b], mode=mode)), signal.convolve(a, b, mode=mode))

    def test_invalid_dims(self):
        assert_raises(ValueError, convolve2d, 3, 4)
        assert_raises(ValueError, convolve2d, [3], [4])
        assert_raises(ValueError, convolve2d, [[[3]]], [[[4]]])

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    def test_large_array(self):
        n = 2 ** 31 // (1000 * np.int64().itemsize)
        _testutils.check_free_memory(2 * n * 1001 * np.int64().itemsize / 1000000.0)
        a = np.zeros(1001 * n, dtype=np.int64)
        a[::2] = 1
        a = np.lib.stride_tricks.as_strided(a, shape=(n, 1000), strides=(8008, 8))
        count = signal.convolve2d(a, [[1, 1]])
        fails = np.where(count > 1)
        assert fails[0].size == 0