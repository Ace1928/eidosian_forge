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
@pytest.mark.parametrize('ascending', [True, False])
def test_sort_values_with_one_partition(ascending):
    modin_df, pandas_df = create_test_dfs(np.array([['hello', 'goodbye'], ['hello', 'Hello']]))
    if StorageFormat.get() == 'Pandas':
        assert modin_df._query_compiler._modin_frame._partitions.shape == (1, 1)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=1, ascending=ascending))