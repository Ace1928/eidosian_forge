from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
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