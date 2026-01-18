import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.skipif(StorageFormat.get() == 'Hdk', reason='https://github.com/modin-project/modin/issues/3941')
@pytest.mark.parametrize('ignore_index', [True, False])
def test_sort_values_preserve_index_names(ignore_index):
    modin_df, pandas_df = create_test_dfs(np.random.choice(128, 128, replace=False).reshape((128, 1)))
    pandas_df.index.names, pandas_df.columns.names = (['custom_name'], ['custom_name'])
    modin_df.index.names, modin_df.columns.names = (['custom_name'], ['custom_name'])
    modin_df.index = modin_df.index
    modin_df.columns = modin_df.columns

    def comparator(df1, df2):
        assert df1.index.names == df2.index.names
        assert df1.columns.names == df2.columns.names
        df_equals(df1, df2)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(df.columns[0], ignore_index=ignore_index), comparator=comparator)