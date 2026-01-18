from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fftpack import ifft, fft, fftn, ifftn, rfft, irfft, fft2
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.xfail(run=False, reason=reason)
def test_notice(self):
    pass