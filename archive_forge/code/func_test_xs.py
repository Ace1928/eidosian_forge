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
def test_xs():
    data = {'num_legs': [4, 4, 2, 2], 'num_wings': [0, 0, 2, 2], 'class': ['mammal', 'mammal', 'mammal', 'bird'], 'animal': ['cat', 'dog', 'bat', 'penguin'], 'locomotion': ['walks', 'walks', 'flies', 'walks']}
    modin_df, pandas_df = create_test_dfs(data)

    def prepare_dataframes(df):
        df = (pd if isinstance(df, pd.DataFrame) else pandas).concat([df, df], axis=0)
        df = df.reset_index(drop=True)
        df = df.join(df, rsuffix='_y')
        return df.set_index(['class', 'animal', 'locomotion'])
    modin_df = prepare_dataframes(modin_df)
    pandas_df = prepare_dataframes(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.xs('mammal'))
    eval_general(modin_df, pandas_df, lambda df: df.xs('cat', level=1))
    eval_general(modin_df, pandas_df, lambda df: df.xs('num_legs', axis=1))
    eval_general(modin_df, pandas_df, lambda df: df.xs('cat', level=1, drop_level=False))
    eval_general(modin_df, pandas_df, lambda df: df.xs(('mammal', 'cat')))
    eval_general(modin_df, pandas_df, lambda df: df.xs(('mammal', 'cat'), drop_level=False))