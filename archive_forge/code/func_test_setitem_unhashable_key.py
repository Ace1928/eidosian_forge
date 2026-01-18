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
def test_setitem_unhashable_key():
    source_modin_df, source_pandas_df = create_test_dfs(test_data['float_nan_data'])
    row_count = source_modin_df.shape[0]

    def _make_copy(df1, df2):
        return (df1.copy(deep=True), df2.copy(deep=True))
    for key in (['col1', 'col2'], ['new_col1', 'new_col2']):
        value = [1, 2]
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)
        value = [[1, 2]] * row_count
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)
        df_value = pandas.DataFrame(value, columns=['value_col1', 'value_col2'])
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, df_value, key)
        value = df_value.to_numpy()
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)
        value = df_value['value_col1']
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key[:1], expected_exception=ValueError('Columns must be same length as key'))
        value = df_value.index
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key[:1], expected_exception=ValueError('Columns must be same length as key'))
        value = 3
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)
        eval_setitem(modin_df, pandas_df, df_value[['value_col1']], key, expected_exception=ValueError('Columns must be same length as key'))