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
def test_out_broadcasts(self):
    arr = np.arange(3).reshape(1, 3)
    out = np.empty((5, 4, 3))
    np.add(arr, arr, out=out)
    assert (out == np.arange(3) * 2).all()
    umt.inner1d(arr, arr, out=out)
    assert (out == 5).all()