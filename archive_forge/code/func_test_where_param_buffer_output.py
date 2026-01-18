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
def test_where_param_buffer_output(self):
    a = np.ones(10, np.int64)
    b = np.ones(10, np.int64)
    c = 1.5 * np.ones(10, np.float64)
    np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])