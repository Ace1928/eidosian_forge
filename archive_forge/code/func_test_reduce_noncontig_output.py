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
def test_reduce_noncontig_output(self):
    x = np.arange(7 * 13 * 8, dtype=np.int16).reshape(7, 13, 8)
    x = x[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
    y_base = np.arange(4 * 4, dtype=np.int16).reshape(4, 4)
    y = y_base[::2, :]
    y_base_copy = y_base.copy()
    r0 = np.add.reduce(x, out=y.copy(), axis=2)
    r1 = np.add.reduce(x, out=y, axis=2)
    assert_equal(r0, r1)
    assert_equal(y_base[1, :], y_base_copy[1, :])
    assert_equal(y_base[3, :], y_base_copy[3, :])