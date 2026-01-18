import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_type_add(self):
    for k, atype in enumerate(types):
        a_scalar = atype(3)
        a_array = np.array([3], dtype=atype)
        for l, btype in enumerate(types):
            b_scalar = btype(1)
            b_array = np.array([1], dtype=btype)
            c_scalar = a_scalar + b_scalar
            c_array = a_array + b_array
            assert_equal(c_scalar.dtype, c_array.dtype, "error with types (%d/'%c' + %d/'%c')" % (k, np.dtype(atype).char, l, np.dtype(btype).char))