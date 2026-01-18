import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.parametrize('dtype', np.typecodes['UnsignedInteger'])
@pytest.mark.parametrize('py_comp_func, np_comp_func', [(operator.lt, np.less), (operator.le, np.less_equal), (operator.gt, np.greater), (operator.ge, np.greater_equal), (operator.eq, np.equal), (operator.ne, np.not_equal)])
@pytest.mark.parametrize('flip', [True, False])
def test_unsigned_signed_direct_comparison(self, dtype, py_comp_func, np_comp_func, flip):
    if flip:
        py_comp = lambda x, y: py_comp_func(y, x)
        np_comp = lambda x, y: np_comp_func(y, x)
    else:
        py_comp = py_comp_func
        np_comp = np_comp_func
    arr = np.array([np.iinfo(dtype).max], dtype=dtype)
    expected = py_comp(int(arr[0]), -1)
    assert py_comp(arr, -1) == expected
    assert np_comp(arr, -1) == expected
    scalar = arr[0]
    assert isinstance(scalar, np.integer)
    assert py_comp(scalar, -1) == expected
    assert np_comp(scalar, -1) == expected