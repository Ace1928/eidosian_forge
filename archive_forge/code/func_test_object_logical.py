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
def test_object_logical(self):
    a = np.array([3, None, True, False, 'test', ''], dtype=object)
    assert_equal(np.logical_or(a, None), np.array([x or None for x in a], dtype=object))
    assert_equal(np.logical_or(a, True), np.array([x or True for x in a], dtype=object))
    assert_equal(np.logical_or(a, 12), np.array([x or 12 for x in a], dtype=object))
    assert_equal(np.logical_or(a, 'blah'), np.array([x or 'blah' for x in a], dtype=object))
    assert_equal(np.logical_and(a, None), np.array([x and None for x in a], dtype=object))
    assert_equal(np.logical_and(a, True), np.array([x and True for x in a], dtype=object))
    assert_equal(np.logical_and(a, 12), np.array([x and 12 for x in a], dtype=object))
    assert_equal(np.logical_and(a, 'blah'), np.array([x and 'blah' for x in a], dtype=object))
    assert_equal(np.logical_not(a), np.array([not x for x in a], dtype=object))
    assert_equal(np.logical_or.reduce(a), 3)
    assert_equal(np.logical_and.reduce(a), None)