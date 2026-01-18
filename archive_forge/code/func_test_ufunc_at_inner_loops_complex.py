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
@pytest.mark.parametrize('typecode', np.typecodes['Complex'])
@pytest.mark.parametrize('ufunc', [np.add, np.subtract, np.multiply])
def test_ufunc_at_inner_loops_complex(self, typecode, ufunc):
    a = np.ones(10, dtype=typecode)
    indx = np.concatenate([np.ones(6, dtype=np.intp), np.full(18, 4, dtype=np.intp)])
    value = a.dtype.type(1j)
    ufunc.at(a, indx, value)
    expected = np.ones_like(a)
    if ufunc is np.multiply:
        expected[1] = expected[4] = -1
    else:
        expected[1] += 6 * (value if ufunc is np.add else -value)
        expected[4] += 18 * (value if ufunc is np.add else -value)
    assert_array_equal(a, expected)