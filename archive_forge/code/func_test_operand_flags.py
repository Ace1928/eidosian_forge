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
def test_operand_flags(self):
    a = np.arange(16, dtype='l').reshape(4, 4)
    b = np.arange(9, dtype='l').reshape(3, 3)
    opflag_tests.inplace_add(a[:-1, :-1], b)
    assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7], [14, 16, 18, 11], [12, 13, 14, 15]], dtype='l'))
    a = np.array(0)
    opflag_tests.inplace_add(a, 3)
    assert_equal(a, 3)
    opflag_tests.inplace_add(a, [3, 4])
    assert_equal(a, 10)