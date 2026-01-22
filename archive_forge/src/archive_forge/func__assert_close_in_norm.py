from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def _assert_close_in_norm(x, y, rtol, size, rdt):
    err_msg = f'size: {size}  rdt: {rdt}'
    assert_array_less(np.linalg.norm(x - y), rtol * np.linalg.norm(x), err_msg)