import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('op', [operator.eq, operator.add])
def test_with_list(op):
    arr = SparseArray([0, 1], fill_value=0)
    result = op(arr, [0, 1])
    expected = op(arr, SparseArray([0, 1]))
    tm.assert_sp_array_equal(result, expected)