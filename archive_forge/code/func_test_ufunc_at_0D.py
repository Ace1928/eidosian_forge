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
def test_ufunc_at_0D(self):
    a = np.array(0)
    np.add.at(a, (), 1)
    assert_equal(a, 1)
    assert_raises(IndexError, np.add.at, a, 0, 1)
    assert_raises(IndexError, np.add.at, a, [], 1)