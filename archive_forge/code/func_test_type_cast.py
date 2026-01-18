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
def test_type_cast(self):
    msg = 'type cast'
    a = np.arange(6, dtype='short').reshape((2, 3))
    assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
    msg = 'type cast on one argument'
    a = np.arange(6).reshape((2, 3))
    b = a + 0.1
    assert_array_almost_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)