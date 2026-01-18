from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_random_real(self):
    for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
        x = random([size]).astype(self.rdt)
        y1 = irfft(rfft(x), n=size)
        y2 = rfft(irfft(x, n=size * 2 - 1))
        assert_equal(y1.dtype, self.rdt)
        assert_equal(y2.dtype, self.cdt)
        assert_array_almost_equal(y1, x, decimal=self.ndec, err_msg='size=%d' % size)
        assert_array_almost_equal(y2, x, decimal=self.ndec, err_msg='size=%d' % size)