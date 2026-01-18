import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
def test_set_change_dtype_slice():
    cols = MultiIndex.from_tuples([('1st', 'a'), ('2nd', 'b'), ('3rd', 'c')])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df['2nd'] = df['2nd'] * 2.0
    blocks = df._to_dict_of_blocks()
    assert sorted(blocks.keys()) == ['float64', 'int64']
    tm.assert_frame_equal(blocks['float64'], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2]))
    tm.assert_frame_equal(blocks['int64'], DataFrame([[3], [6]], columns=cols[2:]))