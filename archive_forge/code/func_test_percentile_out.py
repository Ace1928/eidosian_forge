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
def test_percentile_out(self):
    x = np.array([1, 2, 3])
    y = np.zeros((3,))
    p = (1, 2, 3)
    np.percentile(x, p, out=y)
    assert_equal(np.percentile(x, p), y)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.zeros((3, 3))
    np.percentile(x, p, axis=0, out=y)
    assert_equal(np.percentile(x, p, axis=0), y)
    y = np.zeros((3, 2))
    np.percentile(x, p, axis=1, out=y)
    assert_equal(np.percentile(x, p, axis=1), y)
    x = np.arange(12).reshape(3, 4)
    r0 = np.array([[2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]])
    out = np.empty((2, 4))
    assert_equal(np.percentile(x, (25, 50), axis=0, out=out), r0)
    assert_equal(out, r0)
    r1 = np.array([[0.75, 4.75, 8.75], [1.5, 5.5, 9.5]])
    out = np.empty((2, 3))
    assert_equal(np.percentile(x, (25, 50), axis=1, out=out), r1)
    assert_equal(out, r1)
    r0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    out = np.empty((2, 4), dtype=x.dtype)
    c = np.percentile(x, (25, 50), method='lower', axis=0, out=out)
    assert_equal(c, r0)
    assert_equal(out, r0)
    r1 = np.array([[0, 4, 8], [1, 5, 9]])
    out = np.empty((2, 3), dtype=x.dtype)
    c = np.percentile(x, (25, 50), method='lower', axis=1, out=out)
    assert_equal(c, r1)
    assert_equal(out, r1)