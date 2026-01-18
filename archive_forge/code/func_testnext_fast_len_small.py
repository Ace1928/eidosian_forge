from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
def testnext_fast_len_small(self):
    hams = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15, 16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000}
    for x, y in hams.items():
        assert_equal(next_fast_len(x, True), y)