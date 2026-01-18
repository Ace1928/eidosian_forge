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
@pytest.mark.parametrize('by', [[1, 2, 1, 2], lambda x: x % 3, 'col1', ['col1'], 'col2', ['col2'], pytest.param(['col1', 'col2'], marks=pytest.mark.xfail(reason='Excluded because of bug #1554')), pytest.param(['col2', 'col4'], marks=pytest.mark.xfail(reason='Excluded because of bug #1554')), pytest.param(['col4', 'col2'], marks=pytest.mark.xfail(reason='Excluded because of bug #1554')), pytest.param(['col3', 'col4', 'col2'], marks=pytest.mark.xfail(reason='Excluded because of bug #1554')), ['col5'], ['col1', 'col5'], ['col5', 'col4'], ['col4', 'col5'], ['col5', 'col4', 'col1'], ['col1', pd.Series([1, 5, 7, 8])], [pd.Series([1, 5, 7, 8])], [pd.Series([1, 5, 7, 8]), pd.Series([1, 5, 7, 8]), pd.Series([1, 5, 7, 8]), pd.Series([1, 5, 7, 8]), pd.Series([1, 5, 7, 8])], ['col1', GetColumn('col5')], [GetColumn('col1'), GetColumn('col5')], [GetColumn('col1')]])
@pytest.mark.parametrize('as_index', [True, False], ids=lambda v: f'as_index={v}')
@pytest.mark.parametrize('col1_category', [True, False], ids=lambda v: f'col1_category={v}')
def test_simple_row_groupby(by, as_index, col1_category):
    pandas_df = pandas.DataFrame({'col1': [0, 1, 2, 3], 'col2': [4, 5, np.NaN, 7], 'col3': [np.NaN, np.NaN, 12, 10], 'col4': [17, 13, 16, 15], 'col5': [-4, -5, -6, -7]})
    if col1_category:
        pandas_df = pandas_df.astype({'col1': 'category'})
        pandas_df['col1'] = pandas_df['col1'].cat.as_ordered()
    modin_df = from_pandas(pandas_df)
    n = 1

    def maybe_get_columns(df, by):
        if isinstance(by, list):
            return [o(df) if isinstance(o, GetColumn) else o for o in by]
        else:
            return by
    modin_groupby = modin_df.groupby(by=maybe_get_columns(modin_df, by), as_index=as_index)
    pandas_by = maybe_get_columns(pandas_df, try_cast_to_pandas(by))
    pandas_groupby = pandas_df.groupby(by=pandas_by, as_index=as_index)
    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_shift(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill())
    if as_index:
        eval_general(modin_groupby, pandas_groupby, lambda df: df.nth(0))
    elif not isinstance(pandas_by, list) or len(pandas_by) <= 1:
        eval_general(modin_groupby, pandas_groupby, lambda df: df.nth(0).sort_values('col1').reset_index(drop=True))
    expected_exception = None
    if col1_category:
        expected_exception = TypeError("category dtype does not support aggregation 'sem'")
    eval_general(modin_groupby, pandas_groupby, lambda df: df.sem(), modin_df_almost_equals_pandas, expected_exception=expected_exception)
    eval_mean(modin_groupby, pandas_groupby, numeric_only=True)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax(), expected_exception=False)
    eval_ndim(modin_groupby, pandas_groupby)
    if not check_df_columns_have_nans(modin_df, by):
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support cumsum operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cumsum(), expected_exception=expected_exception)
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support cummax operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cummax(), expected_exception=expected_exception)
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support cummin operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cummin(), expected_exception=expected_exception)
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support cumprod operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cumprod(), expected_exception=expected_exception)
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support cumcount operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cumcount(), expected_exception=expected_exception)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.pct_change(periods=2, fill_method='bfill', limit=1, freq=None, axis=1), modin_df_almost_equals_pandas)
    apply_functions = [lambda df: df.sum(numeric_only=True), lambda df: pandas.Series([1, 2, 3, 4], name='result'), min]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)
    eval_dtypes(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.first())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(), expected_exception=False)
    expected_exception = None
    if col1_category:
        expected_exception = TypeError('category type does not support prod operations')
    eval_general(modin_groupby, pandas_groupby, lambda grp: grp.prod(), expected_exception=expected_exception)
    if as_index:
        eval_std(modin_groupby, pandas_groupby, numeric_only=True)
        eval_var(modin_groupby, pandas_groupby, numeric_only=True)
        eval_skew(modin_groupby, pandas_groupby, numeric_only=True)
    agg_functions = [lambda df: df.sum(), 'min', 'max', min, sum, {'col1': 'count', 'col2': 'count'}, {'col1': 'nunique', 'col2': 'nunique'}]
    for func in agg_functions:
        is_pandas_bug_case = not as_index and col1_category and isinstance(func, dict)
        expected_exception = None
        if col1_category:
            expected_exception = False
        if not is_pandas_bug_case:
            eval_general(modin_groupby, pandas_groupby, lambda grp: grp.agg(func), expected_exception=expected_exception)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.last())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.rank())
    eval_max(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    expected_exception = None
    if col1_category:
        expected_exception = TypeError('category type does not support sum operations')
    eval_general(modin_groupby, pandas_groupby, lambda df: df.sum(), expected_exception=expected_exception)
    eval_ngroup(modin_groupby, pandas_groupby)
    if not (col1_category and (not as_index)):
        eval_general(modin_groupby, pandas_groupby, lambda df: df.nunique())
    expected_exception = None
    if col1_category:
        expected_exception = TypeError("category dtype does not support aggregation 'median'")
    eval_general(modin_groupby, pandas_groupby, lambda df: df.median(), modin_df_almost_equals_pandas, expected_exception=expected_exception)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.head(n))
    eval_general(modin_groupby, pandas_groupby, lambda df: df.cov(), modin_df_almost_equals_pandas)
    if not check_df_columns_have_nans(modin_df, by):
        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for idx, func in enumerate(transform_functions):
            expected_exception = None
            if col1_category:
                if idx == 0:
                    expected_exception = TypeError("unsupported operand type(s) for +: 'Categorical' and 'int'")
                elif idx == 1:
                    expected_exception = TypeError("bad operand type for unary -: 'Categorical'")
            eval_general(modin_groupby, pandas_groupby, lambda df: df.transform(func), expected_exception=expected_exception)
    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        expected_exception = None
        if col1_category:
            expected_exception = TypeError('category type does not support sum operations')
        eval_general(modin_groupby, pandas_groupby, lambda df: df.pipe(func), expected_exception=expected_exception)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.corr(), modin_df_almost_equals_pandas)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    if get_current_execution() != 'BaseOnPython':
        eval_general(modin_groupby, pandas_groupby, lambda df: df.size())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.tail(n))
    eval_quantile(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take([0]))
    if isinstance(by, list) and (not any((isinstance(o, (pd.Series, pandas.Series)) for o in by))):
        eval___getattr__(modin_groupby, pandas_groupby, 'col3')
        eval___getitem__(modin_groupby, pandas_groupby, 'col3', expected_exception=False)
    eval_groups(modin_groupby, pandas_groupby)
    non_by_cols = [col for col in pandas_df.columns[1:] if col not in modin_groupby._internal_by] if isinstance(by, list) else ['col3', 'col4']
    eval___getitem__(modin_groupby, pandas_groupby, non_by_cols, expected_exception=False)
    if len(modin_groupby._internal_by) != 0:
        if not isinstance(by, list):
            by = [by]
        by_from_workaround = [modin_df[getattr(col, 'name', col)].copy() if hashable(col) and col in modin_groupby._internal_by or isinstance(col, GetColumn) else col for col in by]
        modin_groupby = modin_df.groupby(maybe_get_columns(modin_df, by_from_workaround), as_index=True)
        pandas_groupby = pandas_df.groupby(pandas_by, as_index=True)
        eval___getitem__(modin_groupby, pandas_groupby, list(modin_groupby._internal_by) + non_by_cols[:1])