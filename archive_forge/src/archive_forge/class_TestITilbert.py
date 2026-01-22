from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
class TestITilbert:

    def test_definition(self):
        for h in [0.1, 0.5, 1, 5.5, 10]:
            for n in [16, 17, 64, 127]:
                x = arange(n) * 2 * pi / n
                y = itilbert(sin(x), h)
                y1 = direct_itilbert(sin(x), h)
                assert_array_almost_equal(y, y1)
                assert_array_almost_equal(itilbert(sin(x), h), direct_itilbert(sin(x), h))
                assert_array_almost_equal(itilbert(sin(2 * x), h), direct_itilbert(sin(2 * x), h))