from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.skipif(np.longdouble is np.float64, reason='Long double is aliased to double')
class TestIRFFTLongDouble(_TestIRFFTBase):

    def setup_method(self):
        self.cdt = np.complex128
        self.rdt = np.float64
        self.ndec = 14