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
def test_reduce_output_does_not_broadcast_input(self):
    a = np.ones((1, 10))
    out_correct = np.empty((1, 1))
    out_incorrect = np.empty((3, 1))
    np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
    np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
    with assert_raises(ValueError):
        np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
    with assert_raises(ValueError):
        np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)