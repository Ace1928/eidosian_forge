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
def test_where_param(self):
    a = np.arange(7)
    b = np.ones(7)
    c = np.zeros(7)
    np.add(a, b, out=c, where=a % 2 == 1)
    assert_equal(c, [0, 2, 0, 4, 0, 6, 0])
    a = np.arange(4).reshape(2, 2) + 2
    np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
    assert_equal(a, [[2, 27], [16, 5]])
    np.subtract(a, 2, out=a, where=[True, False])
    assert_equal(a, [[0, 27], [14, 5]])