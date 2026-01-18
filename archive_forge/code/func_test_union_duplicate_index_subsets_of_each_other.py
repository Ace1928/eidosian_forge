from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_duplicate_index_subsets_of_each_other(any_dtype_for_small_pos_integer_indexes):
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 2, 2, 3], dtype=dtype)
    b = Index([3, 3, 4], dtype=dtype)
    expected = Index([1, 2, 2, 3, 3, 4], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)