import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_shape_monster(self):
    """Test some more complicated cases that shouldn't be equal"""
    assert_dtype_not_equal(np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))), np.dtype(([('a', 'f4', (1, 2)), ('b', 'f8', (1, 3))], (2, 2))))
    assert_dtype_not_equal(np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))), np.dtype(([('a', 'f4', (2, 1)), ('b', 'i8', (1, 3))], (2, 2))))
    assert_dtype_not_equal(np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))), np.dtype(([('e', 'f8', (1, 3)), ('d', 'f4', (2, 1))], (2, 2))))
    assert_dtype_not_equal(np.dtype(([('a', [('a', 'i4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))), np.dtype(([('a', [('a', 'u4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))))