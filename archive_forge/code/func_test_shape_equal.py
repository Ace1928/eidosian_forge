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
def test_shape_equal(self):
    """Test some data types that are equal"""
    assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', tuple())))
    with pytest.warns(FutureWarning):
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', 1)))
    assert_dtype_equal(np.dtype((int, 2)), np.dtype((int, (2,))))
    assert_dtype_equal(np.dtype(('<f4', (3, 2))), np.dtype(('<f4', (3, 2))))
    d = ([('a', 'f4', (1, 2)), ('b', 'f8', (3, 1))], (3, 2))
    assert_dtype_equal(np.dtype(d), np.dtype(d))