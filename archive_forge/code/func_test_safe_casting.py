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
def test_safe_casting(self):
    a = np.array([1, 2, 3], dtype=int)
    assert_array_equal(assert_no_warnings(np.add, a, 1.1), [2.1, 3.1, 4.1])
    assert_raises(TypeError, np.add, a, 1.1, out=a)

    def add_inplace(a, b):
        a += b
    assert_raises(TypeError, add_inplace, a, 1.1)
    assert_no_warnings(np.add, a, 1.1, out=a, casting='unsafe')
    assert_array_equal(a, [2, 3, 4])