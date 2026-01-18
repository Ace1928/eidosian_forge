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
def test_object_array_accumulate_failure(self):
    res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
    assert_array_equal(res, np.array([1, 1, 3], dtype=object))
    with pytest.raises(TypeError):
        np.add.accumulate([1, None, 2])