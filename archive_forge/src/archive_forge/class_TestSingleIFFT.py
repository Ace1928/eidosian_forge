from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
class TestSingleIFFT(_TestIFFTBase):

    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32
        self.rtol = 1e-05
        self.atol = 0.0001