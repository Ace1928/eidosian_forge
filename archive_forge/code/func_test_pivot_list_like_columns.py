import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input_index, input_columns, input_values, expected_values, expected_columns, expected_index', [('lev4', ['lev3'], 'values', [[0.0, np.nan], [np.nan, 1.0], [2.0, np.nan], [np.nan, 3.0], [4.0, np.nan], [np.nan, 5.0], [6.0, np.nan], [np.nan, 7.0]], Index([1, 2], name='lev3'), Index([1, 2, 3, 4, 5, 6, 7, 8], name='lev4')), (['lev1', 'lev2'], ['lev3'], 'values', [[0, 1], [2, 3], [4, 5], [6, 7]], Index([1, 2], name='lev3'), MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2)], names=['lev1', 'lev2'])), (['lev1'], ['lev2', 'lev3'], 'values', [[0, 1, 2, 3], [4, 5, 6, 7]], MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2)], names=['lev2', 'lev3']), Index([1, 2], name='lev1')), (['lev1', 'lev2'], ['lev3', 'lev4'], 'values', [[0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0]], MultiIndex.from_tuples([(1, 1), (2, 2), (1, 3), (2, 4), (1, 5), (2, 6), (1, 7), (2, 8)], names=['lev3', 'lev4']), MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2)], names=['lev1', 'lev2']))])
def test_pivot_list_like_columns(input_index, input_columns, input_values, expected_values, expected_columns, expected_index):
    df = pd.DataFrame({'lev1': [1, 1, 1, 1, 2, 2, 2, 2], 'lev2': [1, 1, 2, 2, 1, 1, 2, 2], 'lev3': [1, 2, 1, 2, 1, 2, 1, 2], 'lev4': [1, 2, 3, 4, 5, 6, 7, 8], 'values': [0, 1, 2, 3, 4, 5, 6, 7]})
    result = df.pivot(index=input_index, columns=input_columns, values=input_values)
    expected = pd.DataFrame(expected_values, columns=expected_columns, index=expected_index)
    tm.assert_frame_equal(result, expected)