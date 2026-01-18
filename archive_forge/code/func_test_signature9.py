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
def test_signature9(self):
    enabled, num_dims, ixs, flags, sizes = umt.test_signature(1, 1, '(  3)  -> ( )')
    assert_equal(enabled, 1)
    assert_equal(num_dims, (1, 0))
    assert_equal(ixs, (0,))
    assert_equal(flags, (0,))
    assert_equal(sizes, (3,))