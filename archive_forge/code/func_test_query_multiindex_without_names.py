import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_query_multiindex_without_names():

    def make_df(without_index):
        new_df = without_index.set_index(['col1', 'col3'])
        new_df.index.names = [None, None]
        return new_df
    eval_general(*(make_df(df) for df in create_test_dfs(test_data['int_data'])), lambda df: df.query('ilevel_0 % 2 == 0 | ilevel_1 % 2 == 1 | col4 % 2 == 1'))