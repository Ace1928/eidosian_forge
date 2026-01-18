import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('modify_config', [{RangePartitioning: True, IsRayCluster: True}, {RangePartitioning: True, IsRayCluster: False}], indirect=True)
def test_shape_changing_udf(modify_config):
    modin_df, pandas_df = create_test_dfs({'by_col1': [1] * 50 + [10] * 50, 'col2': np.arange(100), 'col3': np.arange(100)})

    def func1(group):
        return pandas.Series([1, 2, 3, 4], index=['new_col1', 'new_col2', 'new_col4', 'new_col3'])
    eval_general(modin_df.groupby('by_col1'), pandas_df.groupby('by_col1'), lambda df: df.apply(func1))

    def func2(group):
        if group.iloc[0, 0] == 1:
            return pandas.Series([1, 2, 3, 4], index=['new_col1', 'new_col2', 'new_col4', 'new_col3']).to_frame().T
        return pandas.Series([20, 33, 44], index=['new_col2', 'new_col3', 'new_col4']).to_frame().T
    eval_general(modin_df.groupby('by_col1'), pandas_df.groupby('by_col1'), lambda df: df.apply(func2))

    def func3(group):
        if group.iloc[0, 0] == 1:
            return pandas.DataFrame([[1, 2, 3]], index=['col1', 'col2', 'col3'])
        return pandas.DataFrame(columns=['col2', 'col3', 'col4', 'col5'])
    eval_general(modin_df.groupby('by_col1'), pandas_df.groupby('by_col1'), lambda df: df.apply(func3))