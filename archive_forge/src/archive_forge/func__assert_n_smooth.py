from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
def _assert_n_smooth(x, n):
    x_orig = x
    if n < 2:
        assert False
    while True:
        q, r = divmod(x, 2)
        if r != 0:
            break
        x = q
    for d in range(3, n + 1, 2):
        while True:
            q, r = divmod(x, d)
            if r != 0:
                break
            x = q
    assert x == 1, f'x={x_orig} is not {n}-smooth, remainder={x}'