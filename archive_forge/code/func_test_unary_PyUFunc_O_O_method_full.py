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
@pytest.mark.parametrize('ufunc', UNARY_OBJECT_UFUNCS)
def test_unary_PyUFunc_O_O_method_full(self, ufunc):
    """Compare the result of the object loop with non-object one"""
    val = np.float64(np.pi / 4)

    class MyFloat(np.float64):

        def __getattr__(self, attr):
            try:
                return super().__getattr__(attr)
            except AttributeError:
                return lambda: getattr(np.core.umath, attr)(val)
    num_arr = np.array(val, dtype=np.float64)
    obj_arr = np.array(MyFloat(val), dtype='O')
    with np.errstate(all='raise'):
        try:
            res_num = ufunc(num_arr)
        except Exception as exc:
            with assert_raises(type(exc)):
                ufunc(obj_arr)
        else:
            res_obj = ufunc(obj_arr)
            assert_array_almost_equal(res_num.astype('O'), res_obj)