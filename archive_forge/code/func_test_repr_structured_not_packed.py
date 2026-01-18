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
def test_repr_structured_not_packed(self):
    dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'], 'formats': ['<u4', 'u1', 'u1', 'u1'], 'offsets': [0, 0, 1, 2], 'titles': ['Color', 'Red pixel', 'Green pixel', 'Blue pixel']}, align=True)
    assert_equal(repr(dt), "dtype({'names': ['rgba', 'r', 'g', 'b'], 'formats': ['<u4', 'u1', 'u1', 'u1'], 'offsets': [0, 0, 1, 2], 'titles': ['Color', 'Red pixel', 'Green pixel', 'Blue pixel'], 'itemsize': 4}, align=True)")
    dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'offsets': [0, 2], 'titles': ['Red pixel', 'Blue pixel'], 'itemsize': 4})
    assert_equal(repr(dt), "dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'offsets': [0, 2], 'titles': ['Red pixel', 'Blue pixel'], 'itemsize': 4})")