import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
@skip_if_array_api_backend('torch')
@array_api_compatible
def test_rfft2(self, xp):
    x = xp.asarray(random((30, 20)))
    expect = fft.fft2(x)[:, :11]
    xp_assert_close(fft.rfft2(x), expect)
    xp_assert_close(fft.rfft2(x, norm='backward'), expect)
    xp_assert_close(fft.rfft2(x, norm='ortho'), expect / xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
    xp_assert_close(fft.rfft2(x, norm='forward'), expect / (30 * 20))