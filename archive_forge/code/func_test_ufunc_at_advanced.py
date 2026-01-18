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
def test_ufunc_at_advanced(self):
    orig = np.arange(4)
    a = orig[:, None][:, 0:0]
    np.add.at(a, [0, 1], 3)
    assert_array_equal(orig, np.arange(4))
    index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
    values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
    np.add.at(values, index, 3)
    assert_array_equal(values, [1, 8, 6, 4])
    values = np.array(['a', 1], dtype=object)
    assert_raises(TypeError, np.add.at, values, [0, 1], 1)
    assert_array_equal(values, np.array(['a', 1], dtype=object))
    assert_raises(ValueError, np.modf.at, np.arange(10), [1])
    a = np.array([1, 2, 3])
    np.maximum.at(a, [0], 0)
    assert_equal(a, np.array([1, 2, 3]))