import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_output_argument(self):
    msg = 'output argument'
    a = np.arange(12).reshape((2, 3, 2))
    b = np.arange(4).reshape((2, 1, 2)) + 1
    c = np.zeros((2, 3), dtype='int')
    umt.inner1d(a, b, c)
    assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
    c[:] = -1
    umt.inner1d(a, b, out=c)
    assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
    msg = 'output argument with type cast'
    c = np.zeros((2, 3), dtype='int16')
    umt.inner1d(a, b, c)
    assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
    c[:] = -1
    umt.inner1d(a, b, out=c)
    assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
    msg = 'output argument with incontiguous layout'
    c = np.zeros((2, 3, 4), dtype='int16')
    umt.inner1d(a, b, c[..., 0])
    assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)
    c[:] = -1
    umt.inner1d(a, b, out=c[..., 0])
    assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)