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
def test_object_array_reduction(self):
    a = np.array(['a', 'b', 'c'], dtype=object)
    assert_equal(np.sum(a), 'abc')
    assert_equal(np.max(a), 'c')
    assert_equal(np.min(a), 'a')
    a = np.array([True, False, True], dtype=object)
    assert_equal(np.sum(a), 2)
    assert_equal(np.prod(a), 0)
    assert_equal(np.any(a), True)
    assert_equal(np.all(a), False)
    assert_equal(np.max(a), True)
    assert_equal(np.min(a), False)
    assert_equal(np.array([[1]], dtype=object).sum(), 1)
    assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
    assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
    assert_equal(np.array([[1], [2, 3]], dtype=object).sum(initial=[0], where=[False, True]), [0, 2, 3])