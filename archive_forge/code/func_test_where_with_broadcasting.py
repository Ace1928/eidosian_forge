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
def test_where_with_broadcasting(self):
    a = np.random.random((5000, 4))
    b = np.random.random((5000, 1))
    where = a > 0.3
    out = np.full_like(a, 0)
    np.less(a, b, where=where, out=out)
    b_where = np.broadcast_to(b, a.shape)[where]
    assert_array_equal(a[where] < b_where, out[where].astype(bool))
    assert not out[~where].any()