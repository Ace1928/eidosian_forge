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
def test_signature_all_None(self):
    res1 = np.add([3], [4], sig=(None, None, None))
    res2 = np.add([3], [4])
    assert_array_equal(res1, res2)
    res1 = np.maximum([3], [4], sig=(None, None, None))
    res2 = np.maximum([3], [4])
    assert_array_equal(res1, res2)
    with pytest.raises(TypeError):
        np.add(3, 4, signature=(None,))