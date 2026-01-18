import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('source, neg_target, abs_target', [([1, 2, 3], [-1, -2, -3], [1, 2, 3]), ([1, 2, None], [-1, -2, None], [1, 2, None]), ([-1, 0, 1], [1, 0, -1], [1, 0, 1])])
def test_unary_int_operators(any_signed_int_ea_dtype, source, neg_target, abs_target):
    dtype = any_signed_int_ea_dtype
    arr = pd.array(source, dtype=dtype)
    neg_result, pos_result, abs_result = (-arr, +arr, abs(arr))
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)
    tm.assert_extension_array_equal(neg_result, neg_target)
    tm.assert_extension_array_equal(pos_result, arr)
    assert not tm.shares_memory(pos_result, arr)
    tm.assert_extension_array_equal(abs_result, abs_target)