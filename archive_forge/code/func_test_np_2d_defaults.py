from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal, assert_array_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
def test_np_2d_defaults(self):
    x = np.arange(0, 1, 0.1).reshape(5, 2)
    shape = None
    axes = None
    shape_expected = np.array([5, 2])
    axes_expected = np.array([0, 1])
    shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    assert_equal(shape_res, shape_expected)
    assert_equal(axes_res, axes_expected)