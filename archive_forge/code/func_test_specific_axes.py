import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_specific_axes(self):
    v = [[1, 1], [3, 4]]
    x = np.array(v)
    dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
    assert_array_equal(gradient(x, axis=0), dx[0])
    assert_array_equal(gradient(x, axis=1), dx[1])
    assert_array_equal(gradient(x, axis=-1), dx[1])
    assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])
    assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
    assert_almost_equal(gradient(x, axis=None), gradient(x))
    assert_array_equal(gradient(x, 2, 3, axis=(1, 0)), [dx[1] / 2.0, dx[0] / 3.0])
    assert_raises(TypeError, gradient, x, 1, 2, axis=1)
    assert_raises(np.AxisError, gradient, x, axis=3)
    assert_raises(np.AxisError, gradient, x, axis=-3)