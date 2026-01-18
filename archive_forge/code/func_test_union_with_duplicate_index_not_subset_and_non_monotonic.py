from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_with_duplicate_index_not_subset_and_non_monotonic(any_dtype_for_small_pos_integer_indexes):
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 2], dtype=dtype)
    b = Index([0, 0, 1], dtype=dtype)
    expected = Index([0, 0, 1, 2], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([0, 0, 1, 2])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = b.union(a)
    tm.assert_index_equal(result, expected)