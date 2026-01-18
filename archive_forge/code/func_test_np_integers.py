from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
def test_np_integers(self):
    ITYPES = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
    for ityp in ITYPES:
        x = ityp(12345)
        testN = next_fast_len(x)
        assert_equal(testN, next_fast_len(int(x)))