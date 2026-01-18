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
def test_ufunc_custom_out(self):
    a = np.array([0, 1, 2], dtype='i8')
    b = np.array([0, 1, 2], dtype='i8')
    c = np.empty(3, dtype=_rational_tests.rational)
    result = _rational_tests.test_add(a, b, c)
    target = np.array([0, 2, 4], dtype=_rational_tests.rational)
    assert_equal(result, target)
    result = _rational_tests.test_add(a, b)
    assert_equal(result, target)
    result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
    assert_equal(result, target)
    with assert_raises(TypeError):
        _rational_tests.test_add(a, np.uint16(2))