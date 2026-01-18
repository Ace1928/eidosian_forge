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
def test_resolve_dtypes_basic(self):
    i4 = np.dtype('i4')
    f4 = np.dtype('f4')
    f8 = np.dtype('f8')
    r = np.add.resolve_dtypes((i4, f4, None))
    assert r == (f8, f8, f8)
    r = np.add.resolve_dtypes((i4, i4, None), signature=(None, None, 'f4'))
    assert r == (f4, f4, f4)
    r = np.add.resolve_dtypes((f4, int, None))
    assert r == (f4, f4, f4)
    with pytest.raises(TypeError):
        np.add.resolve_dtypes((i4, f4, None), casting='no')