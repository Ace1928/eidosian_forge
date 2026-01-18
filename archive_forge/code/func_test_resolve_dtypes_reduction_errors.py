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
def test_resolve_dtypes_reduction_errors(self):
    i2 = np.dtype('i2')
    with pytest.raises(TypeError):
        np.add.resolve_dtypes((None, i2, i2))
    with pytest.raises(TypeError):
        np.add.signature((None, None, 'i4'))