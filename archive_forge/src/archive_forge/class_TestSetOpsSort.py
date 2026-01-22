from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
class TestSetOpsSort:

    @pytest.mark.parametrize('slice_', [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        idx = Index([1, 0, 2])
        other = idx[slice_]
        tm.assert_index_equal(idx.union(other), idx)
        tm.assert_index_equal(other.union(idx), idx)
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.parametrize('slice_', [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        idx = Index([1, 0, 2])
        other = idx[slice_]
        result = idx.union(other, sort=True)
        expected = Index([0, 1, 2])
        tm.assert_index_equal(result, expected)