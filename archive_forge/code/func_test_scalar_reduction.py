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
def test_scalar_reduction(self):
    assert_equal(np.sum(3, axis=0), 3)
    assert_equal(np.prod(3.5, axis=0), 3.5)
    assert_equal(np.any(True, axis=0), True)
    assert_equal(np.all(False, axis=0), False)
    assert_equal(np.max(3, axis=0), 3)
    assert_equal(np.min(2.5, axis=0), 2.5)
    assert_equal(np.power.reduce(3), 3)
    assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
    assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
    assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
    assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)
    assert_(type(np.any(0, axis=0)) is np.bool_)

    class MyArray(np.ndarray):
        pass
    a = np.array(1).view(MyArray)
    assert_(type(np.any(a)) is MyArray)