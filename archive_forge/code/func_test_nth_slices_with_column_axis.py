import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('start, stop, expected_values, expected_columns', [(None, None, [0, 1, 2, 3, 4], list('ABCDE')), (None, 1, [0, 3], list('AD')), (None, 9, [0, 1, 2, 3, 4], list('ABCDE')), (None, -1, [0, 1, 3], list('ABD')), (1, None, [1, 2, 4], list('BCE')), (1, -1, [1], list('B')), (-1, None, [2, 4], list('CE')), (-1, 2, [4], list('E'))])
@pytest.mark.parametrize('method', ['call', 'index'])
def test_nth_slices_with_column_axis(start, stop, expected_values, expected_columns, method):
    df = DataFrame([range(5)], columns=[list('ABCDE')])
    gb = df.groupby([5, 5, 5, 6, 6], axis=1)
    result = {'call': lambda start, stop: gb.nth(slice(start, stop)), 'index': lambda start, stop: gb.nth[start:stop]}[method](start, stop)
    expected = DataFrame([expected_values], columns=[expected_columns])
    tm.assert_frame_equal(result, expected)