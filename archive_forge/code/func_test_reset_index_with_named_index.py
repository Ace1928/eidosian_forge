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
@pytest.mark.parametrize('test_async_reset_index', [False, pytest.param(True, marks=pytest.mark.xfail(StorageFormat.get() == 'Hdk', reason='HDK does not store trivial indexes.'))])
@pytest.mark.parametrize('index_levels_names_max_levels', [0, 1, 2])
def test_reset_index_with_named_index(index_levels_names_max_levels, test_async_reset_index):
    modin_df = pd.DataFrame(test_data_values[0])
    pandas_df = pandas.DataFrame(test_data_values[0])
    index_name = tuple([f'name_{j}' for j in range(0, index_levels_names_max_levels)]) if index_levels_names_max_levels > 0 else 'NAME_OF_INDEX'
    modin_df.index.name = pandas_df.index.name = index_name
    df_equals(modin_df, pandas_df)
    if test_async_reset_index:
        modin_df.index = modin_df.index
        modin_df.modin.to_pandas()
        modin_df._query_compiler._modin_frame.set_index_cache(None)
    df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))
    if test_async_reset_index:
        modin_df.index = modin_df.index
        modin_df.modin.to_pandas()
        modin_df._query_compiler._modin_frame.set_index_cache(None)
    modin_df.reset_index(drop=True, inplace=True)
    pandas_df.reset_index(drop=True, inplace=True)
    df_equals(modin_df, pandas_df)
    modin_df = pd.DataFrame(test_data_values[0])
    pandas_df = pandas.DataFrame(test_data_values[0])
    modin_df.index.name = pandas_df.index.name = index_name
    if test_async_reset_index:
        modin_df.index = modin_df.index
        modin_df._to_pandas()
        modin_df._query_compiler._modin_frame.set_index_cache(None)
    df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))