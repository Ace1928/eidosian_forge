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
def test_reducelike_output_needs_identical_cast(self):
    arr = np.ones(20, dtype='f8')
    out = np.empty((), dtype=arr.dtype.newbyteorder())
    expected = np.add.reduce(arr)
    np.add.reduce(arr, out=out)
    assert_array_equal(expected, out)
    out = np.empty(2, dtype=arr.dtype.newbyteorder())
    expected = np.add.reduceat(arr, [0, 1])
    np.add.reduceat(arr, [0, 1], out=out)
    assert_array_equal(expected, out)
    out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
    expected = np.add.accumulate(arr)
    np.add.accumulate(arr, out=out)
    assert_array_equal(expected, out)