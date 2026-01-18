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
@pytest.mark.parametrize(('axis', 'where'), ((0, np.array([True, False, True])), (1, [True, True, False]), (None, True)))
@pytest.mark.parametrize('initial', (-np.inf, 5.0))
def test_reduction_with_where_and_initial(self, axis, where, initial):
    a = np.arange(9.0).reshape(3, 3)
    a_copy = a.copy()
    a_check = np.full(a.shape, -np.inf)
    np.positive(a, out=a_check, where=where)
    res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
    check = a_check.max(axis, initial=initial)
    assert_equal(res, check)