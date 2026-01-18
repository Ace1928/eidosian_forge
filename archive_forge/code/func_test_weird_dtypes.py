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
def test_weird_dtypes(self):
    S0 = np.dtype('S0')
    r = np.equal.resolve_dtypes((S0, S0, None))
    assert r == (S0, S0, np.dtype(bool))
    dts = np.dtype('10i')
    with pytest.raises(TypeError):
        np.equal.resolve_dtypes((dts, dts, None))