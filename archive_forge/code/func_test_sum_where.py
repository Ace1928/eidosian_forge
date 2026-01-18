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
def test_sum_where(self):
    assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], where=[True, False]), 4.0)
    assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0, initial=5.0, where=[True, False]), [9.0, 5.0])