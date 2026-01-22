from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class _TestIRFFTBase:

    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x1 = [1, 2 + 3j, 4 + 1j, 1 + 2j, 3 + 4j]
        x1_1 = [1, 2 + 3j, 4 + 1j, 2 + 3j, 4, 2 - 3j, 4 - 1j, 2 - 3j]
        x1 = x1_1[:5]
        x2_1 = [1, 2 + 3j, 4 + 1j, 2 + 3j, 4 + 5j, 4 - 5j, 2 - 3j, 4 - 1j, 2 - 3j]
        x2 = x2_1[:5]

        def _test(x, xr):
            y = irfft(np.array(x, dtype=self.cdt), n=len(xr))
            y1 = direct_irdft(x, len(xr))
            assert_equal(y.dtype, self.rdt)
            assert_array_almost_equal(y, y1, decimal=self.ndec)
            assert_array_almost_equal(y, ifft(xr), decimal=self.ndec)
        _test(x1, x1_1)
        _test(x2, x2_1)

    def test_djbfft(self):
        for i in range(2, 14):
            n = 2 ** i
            x = np.arange(-1, n, 2) + 1j * np.arange(0, n + 1, 2)
            x[0] = 0
            if n % 2 == 0:
                x[-1] = np.real(x[-1])
            y1 = np.fft.irfft(x)
            y = irfft(x)
            assert_array_almost_equal(y, y1)

    def test_random_real(self):
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = random([size]).astype(self.rdt)
            y1 = irfft(rfft(x), n=size)
            y2 = rfft(irfft(x, n=size * 2 - 1))
            assert_equal(y1.dtype, self.rdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x, decimal=self.ndec, err_msg='size=%d' % size)
            assert_array_almost_equal(y2, x, decimal=self.ndec, err_msg='size=%d' % size)

    def test_size_accuracy(self):
        if self.rdt == np.float32:
            rtol = 1e-05
        elif self.rdt == np.float64:
            rtol = 1e-10
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = irfft(rfft(x), len(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            y = rfft(irfft(x, 2 * len(x) - 1))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

    def test_invalid_sizes(self):
        assert_raises(ValueError, irfft, [])
        assert_raises(ValueError, irfft, [[1, 1], [2, 2]], -5)