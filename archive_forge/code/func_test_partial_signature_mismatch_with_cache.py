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
def test_partial_signature_mismatch_with_cache(self):
    with pytest.raises(TypeError):
        np.add(np.float16(1), np.uint64(2), sig=('e', 'd', None))
    np.add(np.float16(1), np.float64(2))
    with pytest.raises(TypeError):
        np.add(np.float16(1), np.uint64(2), sig=('e', 'd', None))