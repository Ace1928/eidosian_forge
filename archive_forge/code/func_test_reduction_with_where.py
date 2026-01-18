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
@pytest.mark.parametrize('axis', (0, 1, None))
@pytest.mark.parametrize('where', (np.array([False, True, True]), np.array([[True], [False], [True]]), np.array([[True, False, False], [False, True, False], [False, True, True]])))
def test_reduction_with_where(self, axis, where):
    a = np.arange(9.0).reshape(3, 3)
    a_copy = a.copy()
    a_check = np.zeros_like(a)
    np.positive(a, out=a_check, where=where)
    res = np.add.reduce(a, axis=axis, where=where)
    check = a_check.sum(axis)
    assert_equal(res, check)
    assert_array_equal(a, a_copy)