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
def test_signature_errors(self):
    with pytest.raises(TypeError, match='the signature object to ufunc must be a string or'):
        np.add(3, 4, signature=123.0)
    with pytest.raises(ValueError):
        np.add(3, 4, signature='%^->#')
    with pytest.raises(ValueError):
        np.add(3, 4, signature=b'ii-i')
    with pytest.raises(ValueError):
        np.add(3, 4, signature='ii>i')
    with pytest.raises(ValueError):
        np.add(3, 4, signature=(None, 'f8'))
    with pytest.raises(UnicodeDecodeError):
        np.add(3, 4, signature=b'\xff\xff->i')