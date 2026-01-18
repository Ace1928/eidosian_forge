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
@pytest.mark.parametrize('out_shape', [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
    a = np.arange(12.0).reshape(4, 3)
    out = np.empty(out_shape, a.dtype)
    correct_out = f_reduce(a, axis=0, keepdims=keepdims)
    if out_shape != correct_out.shape:
        with assert_raises(ValueError):
            f_reduce(a, axis=0, out=out, keepdims=keepdims)
    else:
        check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
        assert_(check is out)
        assert_array_equal(check, correct_out)