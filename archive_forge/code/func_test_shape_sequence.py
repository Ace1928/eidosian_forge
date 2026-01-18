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
def test_shape_sequence(self):
    a = np.array([1, 2, 3], dtype=np.int16)
    l = [1, 2, 3]
    dt = np.dtype([('a', 'f4', a)])
    assert_(isinstance(dt['a'].shape, tuple))
    assert_(isinstance(dt['a'].shape[0], int))
    dt = np.dtype([('a', 'f4', l)])
    assert_(isinstance(dt['a'].shape, tuple))

    class IntLike:

        def __index__(self):
            return 3

        def __int__(self):
            return 3
    dt = np.dtype([('a', 'f4', IntLike())])
    assert_(isinstance(dt['a'].shape, tuple))
    assert_(isinstance(dt['a'].shape[0], int))
    dt = np.dtype([('a', 'f4', (IntLike(),))])
    assert_(isinstance(dt['a'].shape, tuple))
    assert_(isinstance(dt['a'].shape[0], int))