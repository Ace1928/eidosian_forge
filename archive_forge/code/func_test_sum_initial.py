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
def test_sum_initial(self):
    assert_equal(np.sum([3], initial=2), 5)
    assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)
    assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2), [12, 12, 12])