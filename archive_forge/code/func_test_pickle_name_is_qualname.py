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
@pytest.mark.skipif(IS_PYPY, reason="'is' check does not work on PyPy")
def test_pickle_name_is_qualname(self):
    _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc
    obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
    assert obj is umt._pickleable_module_global_ufunc