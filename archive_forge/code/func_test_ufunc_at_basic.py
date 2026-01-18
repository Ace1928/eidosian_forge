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
@pytest.mark.parametrize('a', (np.arange(10, dtype=int), np.arange(10, dtype=_rational_tests.rational)))
def test_ufunc_at_basic(self, a):
    aa = a.copy()
    np.add.at(aa, [2, 5, 2], 1)
    assert_equal(aa, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])
    with pytest.raises(ValueError):
        np.add.at(aa, [2, 5, 3])
    aa = a.copy()
    np.negative.at(aa, [2, 5, 3])
    assert_equal(aa, [0, 1, -2, -3, 4, -5, 6, 7, 8, 9])
    aa = a.copy()
    b = np.array([100, 100, 100])
    np.add.at(aa, [2, 5, 2], b)
    assert_equal(aa, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])
    with pytest.raises(ValueError):
        np.negative.at(a, [2, 5, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        np.add.at(a, [2, 5, 3], [[1, 2], 1])