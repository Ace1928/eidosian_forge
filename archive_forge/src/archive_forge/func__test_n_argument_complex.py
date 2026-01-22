from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def _test_n_argument_complex(self):
    x1 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
    x2 = np.array([1, 2, 3, 4 + 1j], dtype=self.cdt)
    y = fft([x1, x2], n=4)
    assert_equal(y.dtype, self.cdt)
    assert_equal(y.shape, (2, 4))
    assert_array_almost_equal(y[0], direct_dft(x1))
    assert_array_almost_equal(y[1], direct_dft(x2))