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
@array_api_compatible
@skip_if_array_api_backend('torch')
def test_not_last_axis_success(self, xp):
    ar, ai = np.random.random((2, 16, 8, 32))
    a = ar + 1j * ai
    a = xp.asarray(a)
    axes = (-2,)
    fft.irfftn(a, axes=axes)