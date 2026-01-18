from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_shape_argument(self):
    small_x = [[1, 2, 3], [4, 5, 6]]
    large_x1 = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    y = fftn(small_x, s=(4, 4))
    assert_array_almost_equal(y, fftn(large_x1))
    y = fftn(small_x, s=(3, 4))
    assert_array_almost_equal(y, fftn(large_x1[:-1]))