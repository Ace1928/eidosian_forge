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
@pytest.mark.parametrize('does_value_have_different_columns', [True, False])
def test_setitem_2d_update(does_value_have_different_columns):

    def test(dfs, iloc):
        """Update columns on the given numeric indices."""
        df1, df2 = dfs
        cols1 = df1.columns[iloc].tolist()
        cols2 = df2.columns[iloc].tolist()
        df1[cols1] = df2[cols2]
        return df1
    modin_df, pandas_df = create_test_dfs(test_data['int_data'])
    modin_df2, pandas_df2 = create_test_dfs(test_data['int_data'])
    modin_df2 *= 10
    pandas_df2 *= 10
    if does_value_have_different_columns:
        new_columns = [f'{col}_new' for col in modin_df.columns]
        modin_df2.columns = new_columns
        pandas_df2.columns = new_columns
    modin_dfs = (modin_df, modin_df2)
    pandas_dfs = (pandas_df, pandas_df2)
    eval_general(modin_dfs, pandas_dfs, test, iloc=[0, 1, 2])
    eval_general(modin_dfs, pandas_dfs, test, iloc=[0, -1])
    eval_general(modin_dfs, pandas_dfs, test, iloc=slice(1, None))
    eval_general(modin_dfs, pandas_dfs, test, iloc=slice(None, -2))
    eval_general(modin_dfs, pandas_dfs, test, iloc=[0, 1, 5, 6, 9, 10, -2, -1])
    eval_general(modin_dfs, pandas_dfs, test, iloc=[5, 4, 0, 10, 1, -1])
    eval_general(modin_dfs, pandas_dfs, test, iloc=slice(None, None, 2))