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
def test_overwrite_keyword(self):
    a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])
    a0 = np.array(1)
    a1 = np.arange(2)
    a2 = np.arange(6).reshape(2, 3)
    assert_allclose(np.median(a0.copy(), overwrite_input=True), 1)
    assert_allclose(np.median(a1.copy(), overwrite_input=True), 0.5)
    assert_allclose(np.median(a2.copy(), overwrite_input=True), 2.5)
    assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=0), [1.5, 2.5, 3.5])
    assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=1), [1, 4])
    assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=None), 2.5)
    assert_allclose(np.median(a3.copy(), overwrite_input=True, axis=0), [3, 4])
    assert_allclose(np.median(a3.T.copy(), overwrite_input=True, axis=1), [3, 4])
    a4 = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    np.random.shuffle(a4.ravel())
    assert_allclose(np.median(a4, axis=None), np.median(a4.copy(), axis=None, overwrite_input=True))
    assert_allclose(np.median(a4, axis=0), np.median(a4.copy(), axis=0, overwrite_input=True))
    assert_allclose(np.median(a4, axis=1), np.median(a4.copy(), axis=1, overwrite_input=True))
    assert_allclose(np.median(a4, axis=2), np.median(a4.copy(), axis=2, overwrite_input=True))