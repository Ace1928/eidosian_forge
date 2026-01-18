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
def test_reducelike_byteorder_resolution(self):
    arr_be = np.arange(10, dtype='>i8')
    arr_le = np.arange(10, dtype='<i8')
    assert np.add.reduce(arr_be) == np.add.reduce(arr_le)
    assert_array_equal(np.add.accumulate(arr_be), np.add.accumulate(arr_le))
    assert_array_equal(np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))