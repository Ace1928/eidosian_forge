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
def test_rename_axis():
    data = {'num_legs': [4, 4, 2], 'num_arms': [0, 0, 2]}
    index = ['dog', 'cat', 'monkey']
    modin_df = pd.DataFrame(data, index)
    pandas_df = pandas.DataFrame(data, index)
    df_equals(modin_df.rename_axis('animal'), pandas_df.rename_axis('animal'))
    df_equals(modin_df.rename_axis('limbs', axis='columns'), pandas_df.rename_axis('limbs', axis='columns'))
    modin_df.rename_axis('limbs', axis='columns', inplace=True)
    pandas_df.rename_axis('limbs', axis='columns', inplace=True)
    df_equals(modin_df, pandas_df)
    new_index = pd.MultiIndex.from_product([['mammal'], ['dog', 'cat', 'monkey']], names=['type', 'name'])
    modin_df.index = new_index
    pandas_df.index = new_index
    df_equals(modin_df.rename_axis(index={'type': 'class'}), pandas_df.rename_axis(index={'type': 'class'}))
    df_equals(modin_df.rename_axis(columns=str.upper), pandas_df.rename_axis(columns=str.upper))
    df_equals(modin_df.rename_axis(columns=[str.upper(o) for o in modin_df.columns.names]), pandas_df.rename_axis(columns=[str.upper(o) for o in pandas_df.columns.names]))
    with pytest.raises(ValueError):
        df_equals(modin_df.rename_axis(str.upper, axis=1), pandas_df.rename_axis(str.upper, axis=1))