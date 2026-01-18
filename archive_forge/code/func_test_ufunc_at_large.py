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
def test_ufunc_at_large(self):
    indices = np.zeros(8195, dtype=np.int16)
    b = np.zeros(8195, dtype=float)
    b[0] = 10
    b[1] = 5
    b[8192:] = 100
    a = np.zeros(1, dtype=float)
    np.add.at(a, indices, b)
    assert a[0] == b.sum()