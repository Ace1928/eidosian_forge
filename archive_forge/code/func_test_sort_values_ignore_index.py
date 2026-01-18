import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('original_dict, sorted_dict, ignore_index, output_index', [({'A': [1, 2, 3]}, {'A': [3, 2, 1]}, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [3, 2, 1]}, False, [2, 1, 0]), ({'A': [1, 2, 3], 'B': [2, 3, 4]}, {'A': [3, 2, 1], 'B': [4, 3, 2]}, True, [0, 1, 2]), ({'A': [1, 2, 3], 'B': [2, 3, 4]}, {'A': [3, 2, 1], 'B': [4, 3, 2]}, False, [2, 1, 0])])
def test_sort_values_ignore_index(self, inplace, original_dict, sorted_dict, ignore_index, output_index):
    df = DataFrame(original_dict)
    expected = DataFrame(sorted_dict, index=output_index)
    kwargs = {'ignore_index': ignore_index, 'inplace': inplace}
    if inplace:
        result_df = df.copy()
        result_df.sort_values('A', ascending=False, **kwargs)
    else:
        result_df = df.sort_values('A', ascending=False, **kwargs)
    tm.assert_frame_equal(result_df, expected)
    tm.assert_frame_equal(df, DataFrame(original_dict))