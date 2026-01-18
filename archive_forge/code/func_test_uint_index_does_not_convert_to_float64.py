import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [list, lambda x: np.array(x, dtype=object), lambda x: Index(x, dtype=object)])
def test_uint_index_does_not_convert_to_float64(box):
    series = Series([0, 1, 2, 3, 4, 5], index=[7606741985629028552, 17876870360202815256, 17876870360202815256, 13106359306506049338, 8991270399732411471, 8991270399732411472])
    result = series.loc[box([7606741985629028552, 17876870360202815256])]
    expected = Index([7606741985629028552, 17876870360202815256, 17876870360202815256], dtype='uint64')
    tm.assert_index_equal(result.index, expected)
    tm.assert_equal(result, series.iloc[:3])