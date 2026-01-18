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
@pytest.mark.parametrize('modify_config', [{RangePartitioning: True}], indirect=True)
@pytest.mark.parametrize('observed', [False])
@pytest.mark.parametrize('as_index', [True])
@pytest.mark.parametrize('func', [pytest.param(lambda grp: grp.sum(), id='sum'), pytest.param(lambda grp: grp.size(), id='size'), pytest.param(lambda grp: grp.apply(lambda df: df.sum()), id='apply_sum'), pytest.param(lambda grp: grp.apply(lambda df: df.sum() if len(df) > 0 else pandas.Series([10] * len(df.columns), index=df.columns)), id='apply_transform')])
@pytest.mark.parametrize('by_cols, cat_cols', [('a', ['a']), ('b', ['b']), ('e', ['e']), (['a', 'e'], ['a']), (['a', 'e'], ['e']), (['a', 'e'], ['a', 'e']), (['b', 'e'], ['b']), (['b', 'e'], ['e']), (['b', 'e'], ['b', 'e']), (['a', 'b', 'e'], ['a']), (['a', 'b', 'e'], ['b']), (['a', 'b', 'e'], ['e']), (['a', 'b', 'e'], ['a', 'e']), (['a', 'b', 'e'], ['a', 'b', 'e'])])
@pytest.mark.parametrize('exclude_values', [pytest.param(lambda row: ~row['a'].isin(['a', 'e']), id='exclude_from_a'), pytest.param(lambda row: ~row['b'].isin([4]), id='exclude_from_b'), pytest.param(lambda row: ~row['e'].isin(['x']), id='exclude_from_e'), pytest.param(lambda row: ~row['a'].isin(['a', 'e']) & ~row['b'].isin([4]), id='exclude_from_a_b'), pytest.param(lambda row: ~row['b'].isin([4]) & ~row['e'].isin(['x']), id='exclude_from_b_e'), pytest.param(lambda row: ~row['a'].isin(['a', 'e']) & ~row['b'].isin([4]) & ~row['e'].isin(['x']), id='exclude_from_a_b_e')])
def test_range_groupby_categories(observed, func, by_cols, cat_cols, exclude_values, as_index, modify_config):
    data = {'a': ['a', 'b', 'c', 'd', 'e', 'b', 'g', 'a'] * 32, 'b': [1, 2, 3, 4] * 64, 'c': range(256), 'd': range(256), 'e': ['x', 'y'] * 128}
    md_df, pd_df = create_test_dfs(data)
    md_df = md_df.astype({col: 'category' for col in cat_cols})[exclude_values]
    pd_df = pd_df.astype({col: 'category' for col in cat_cols})[exclude_values]
    md_res = func(md_df.groupby(by_cols, observed=observed, as_index=as_index))
    pd_res = func(pd_df.groupby(by_cols, observed=observed, as_index=as_index))
    df_equals(md_res.sort_index(axis=0), pd_res.sort_index(axis=0))