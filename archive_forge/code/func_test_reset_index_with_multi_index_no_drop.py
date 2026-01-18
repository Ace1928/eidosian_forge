import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('test_async_reset_index', [False, True])
@pytest.mark.parametrize('data', [pytest.param(test_data['int_data'], marks=pytest.mark.exclude_by_default), test_data['float_nan_data']], ids=['int_data', 'float_nan_data'])
@pytest.mark.parametrize('nlevels', [3])
@pytest.mark.parametrize('columns_multiindex', [True, False])
@pytest.mark.parametrize('level', ['no_level', None, 0, 1, 2, [2, 0], [2, 1], [1, 0], pytest.param([2, 1, 2], marks=pytest.mark.exclude_by_default), pytest.param([0, 0, 0, 0], marks=pytest.mark.exclude_by_default), pytest.param(['level_name_1'], marks=pytest.mark.exclude_by_default), pytest.param(['level_name_2', 'level_name_1'], marks=pytest.mark.exclude_by_default), pytest.param([2, 'level_name_0'], marks=pytest.mark.exclude_by_default)])
@pytest.mark.parametrize('col_level', ['no_col_level', 0, 1, 2])
@pytest.mark.parametrize('col_fill', ['no_col_fill', None, 0, 'new'])
@pytest.mark.parametrize('drop', [False])
@pytest.mark.parametrize('multiindex_levels_names_max_levels', [0, 1, 2, pytest.param(3, marks=pytest.mark.exclude_by_default), pytest.param(4, marks=pytest.mark.exclude_by_default)])
@pytest.mark.parametrize('none_in_index_names', [pytest.param(False, marks=pytest.mark.exclude_by_default), True, 'mixed_1st_None', pytest.param('mixed_2nd_None', marks=pytest.mark.exclude_by_default)])
def test_reset_index_with_multi_index_no_drop(data, nlevels, columns_multiindex, level, col_level, col_fill, drop, multiindex_levels_names_max_levels, none_in_index_names, test_async_reset_index):
    data_rows = len(data[list(data.keys())[0]])
    index = generate_multiindex(data_rows, nlevels=nlevels)
    data_columns = len(data.keys())
    columns = generate_multiindex(data_columns, nlevels=nlevels) if columns_multiindex else pandas.RangeIndex(0, data_columns)
    data = {columns[ind]: data[key] for ind, key in enumerate(data)}
    index.names = [f'level_{i}' for i in range(index.nlevels)] if multiindex_levels_names_max_levels == 0 else [tuple([f'level_{i}_name_{j}' for j in range(0, max(multiindex_levels_names_max_levels + 1 - index.nlevels, 0) + i)]) if max(multiindex_levels_names_max_levels + 1 - index.nlevels, 0) + i > 0 else f'level_{i}' for i in range(index.nlevels)]
    if none_in_index_names is True:
        index.names = [None] * len(index.names)
    elif none_in_index_names:
        names_list = list(index.names)
        start_index = 0 if none_in_index_names == 'mixed_1st_None' else 1
        names_list[start_index::2] = [None] * len(names_list[start_index::2])
        index.names = names_list
    modin_df = pd.DataFrame(data, index=index, columns=columns)
    pandas_df = pandas.DataFrame(data, index=index, columns=columns)
    if isinstance(level, list):
        level = [index.names[int(x[len('level_name_'):])] if isinstance(x, str) and x.startswith('level_name_') else x for x in level]
    kwargs = {'drop': drop}
    if level != 'no_level':
        kwargs['level'] = level
    if col_level != 'no_col_level':
        kwargs['col_level'] = col_level
    if col_fill != 'no_col_fill':
        kwargs['col_fill'] = col_fill
    if test_async_reset_index:
        modin_df._query_compiler._modin_frame.set_index_cache(None)
    eval_general(modin_df, pandas_df, lambda df: df.reset_index(**kwargs), comparator_kwargs={'check_dtypes': False})