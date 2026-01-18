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
def test_sort_values_descending_with_only_two_bins():
    part1 = pd.DataFrame({'a': [1, 2, 3, 4]})
    part2 = pd.DataFrame({'a': [5, 6, 7, 8]})
    modin_df = pd.concat([part1, part2])
    pandas_df = modin_df._to_pandas()
    if StorageFormat.get() == 'Pandas':
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by='a', ascending=False))