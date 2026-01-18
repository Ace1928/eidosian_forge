import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [1, 2], 'M2': [3, 4]}, True, True, [0, 1]), ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [2, 1], 'M2': [4, 3]}, False, True, [0, 1]), ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [1, 2], 'M2': [3, 4]}, True, False, MultiIndex.from_tuples([(2, 1), (3, 4)], names=list('AB'))), ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [2, 1], 'M2': [4, 3]}, False, False, MultiIndex.from_tuples([(3, 4), (2, 1)], names=list('AB')))])
def test_sort_index_ignore_index_multi_index(self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index):
    mi = MultiIndex.from_tuples([(2, 1), (3, 4)], names=list('AB'))
    df = DataFrame(original_dict, index=mi)
    expected_df = DataFrame(sorted_dict, index=output_index)
    kwargs = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
    if inplace:
        result_df = df.copy()
        result_df.sort_index(**kwargs)
    else:
        result_df = df.sort_index(**kwargs)
    tm.assert_frame_equal(result_df, expected_df)
    tm.assert_frame_equal(df, DataFrame(original_dict, index=mi))