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
def test_object_array_reduceat_inplace(self):
    arr = np.empty(4, dtype=object)
    arr[:] = [[1] for i in range(4)]
    out = np.empty(4, dtype=object)
    out[:] = [[1] for i in range(4)]
    np.add.reduceat(arr, np.arange(4), out=arr)
    np.add.reduceat(arr, np.arange(4), out=arr)
    assert_array_equal(arr, out)
    arr = np.ones((2, 4), dtype=object)
    arr[0, :] = [[2] for i in range(4)]
    out = np.ones((2, 4), dtype=object)
    out[0, :] = [[2] for i in range(4)]
    np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
    np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
    assert_array_equal(arr, out)