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
@pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'], iter_struct_object_dtypes())
@pytest.mark.parametrize(['shape', 'index', 'items_changed'], [((3,), ([0, 2],), 2), ((3, 2), ([0, 2], slice(None)), 4), ((3, 2), ([0, 2], [1]), 2), ((3,), [True, False, True], 2)])
def test_structured_object_indexing(self, shape, index, items_changed, dt, pat, count, singleton):
    """Structured object reference counting for advanced indexing."""
    val0 = -4
    val1 = -5
    arr = np.full(shape, val0, dt)
    gc.collect()
    before_val0 = sys.getrefcount(val0)
    before_val1 = sys.getrefcount(val1)
    part = arr[index]
    after_val0 = sys.getrefcount(val0)
    assert after_val0 - before_val0 == count * items_changed
    del part
    arr[index] = val1
    gc.collect()
    after_val0 = sys.getrefcount(val0)
    after_val1 = sys.getrefcount(val1)
    assert before_val0 - after_val0 == count * items_changed
    assert after_val1 - before_val1 == count * items_changed