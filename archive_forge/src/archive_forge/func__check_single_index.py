import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def _check_single_index(self, arr, index):
    """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be an arange.
        index : indexing object
            Index being tested. Must be a single index and not a tuple
            of indexing objects (see also `_check_multi_index`).
        """
    try:
        mimic_get, no_copy = self._get_multi_index(arr, (index,))
    except Exception as e:
        if HAS_REFCOUNT:
            prev_refcount = sys.getrefcount(arr)
        assert_raises(type(e), arr.__getitem__, index)
        assert_raises(type(e), arr.__setitem__, index, 0)
        if HAS_REFCOUNT:
            assert_equal(prev_refcount, sys.getrefcount(arr))
        return
    self._compare_index_result(arr, index, mimic_get, no_copy)