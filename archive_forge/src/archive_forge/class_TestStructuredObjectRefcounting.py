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
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='Python 3.12 has immortal refcounts, this test will no longer work. See gh-23986')
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
class TestStructuredObjectRefcounting:
    """These tests cover various uses of complicated structured types which
    include objects and thus require reference counting.
    """

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'], iter_struct_object_dtypes())
    @pytest.mark.parametrize(['creation_func', 'creation_obj'], [pytest.param(np.empty, None, marks=pytest.mark.skip("unreliable due to python's behaviour")), (np.ones, 1), (np.zeros, 0)])
    def test_structured_object_create_delete(self, dt, pat, count, singleton, creation_func, creation_obj):
        """Structured object reference counting in creation and deletion"""
        gc.collect()
        before = sys.getrefcount(creation_obj)
        arr = creation_func(3, dt)
        now = sys.getrefcount(creation_obj)
        assert now - before == count * 3
        del arr
        now = sys.getrefcount(creation_obj)
        assert now == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'], iter_struct_object_dtypes())
    def test_structured_object_item_setting(self, dt, pat, count, singleton):
        """Structured object reference counting for simple item setting"""
        one = 1
        gc.collect()
        before = sys.getrefcount(singleton)
        arr = np.array([pat] * 3, dt)
        assert sys.getrefcount(singleton) - before == count * 3
        before2 = sys.getrefcount(one)
        arr[...] = one
        after2 = sys.getrefcount(one)
        assert after2 - before2 == count * 3
        del arr
        gc.collect()
        assert sys.getrefcount(one) == before2
        assert sys.getrefcount(singleton) == before

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

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'], iter_struct_object_dtypes())
    def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
        """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
        indices = [0, 1]
        arr = np.array([pat] * 3, dt)
        gc.collect()
        before = sys.getrefcount(singleton)
        res = arr.take(indices)
        after = sys.getrefcount(singleton)
        assert after - before == count * 2
        new = res.repeat(10)
        gc.collect()
        after_repeat = sys.getrefcount(singleton)
        assert after_repeat - after == count * 2 * 10