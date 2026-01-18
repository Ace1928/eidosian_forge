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
@pytest.mark.parametrize('index', [pandas.Index([11, 22, 33, 44], name='col0'), pandas.MultiIndex.from_product([[100, 200], [300, 400]], names=['level1', 'col0'])], ids=['index', 'multiindex'])
def test_reset_index_metadata_update(index, test_async_reset_index):
    modin_df, pandas_df = create_test_dfs({'col0': [0, 1, 2, 3]}, index=index)
    modin_df.columns = pandas_df.columns = ['col1']
    if test_async_reset_index:
        modin_df.index = modin_df.index
        modin_df._to_pandas()
        modin_df._query_compiler._modin_frame.set_index_cache(None)
    eval_general(modin_df, pandas_df, lambda df: df.reset_index())