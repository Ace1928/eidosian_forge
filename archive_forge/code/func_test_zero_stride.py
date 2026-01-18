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
def test_zero_stride(self):
    arr = np.ones(1, dtype='i8')
    arr = np.broadcast_to(arr, 10)
    assert arr.strides == (0,)
    with pytest.raises(ValueError):
        arr.dtype = 'i1'