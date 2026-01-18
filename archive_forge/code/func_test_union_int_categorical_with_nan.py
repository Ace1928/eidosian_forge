from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_int_categorical_with_nan():
    ci = CategoricalIndex([1, 2, np.nan])
    assert ci.categories.dtype.kind == 'i'
    idx = Index([1, 2])
    result = idx.union(ci)
    expected = Index([1, 2, np.nan], dtype=np.float64)
    tm.assert_index_equal(result, expected)
    result = ci.union(idx)
    tm.assert_index_equal(result, expected)