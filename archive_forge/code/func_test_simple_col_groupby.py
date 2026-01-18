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
def test_simple_col_groupby():
    pandas_df = pandas.DataFrame({'col1': [0, 3, 2, 3], 'col2': [4, 1, 6, 7], 'col3': [3, 8, 2, 10], 'col4': [1, 13, 6, 15], 'col5': [-4, 5, 6, -7]})
    modin_df = from_pandas(pandas_df)
    by = [1, 2, 3, 2, 1]
    modin_groupby = modin_df.groupby(axis=1, by=by)
    pandas_groupby = pandas_df.groupby(axis=1, by=by)
    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_shift(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.sem(), modin_df_almost_equals_pandas)
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin())
    eval_quantile(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.pct_change(), modin_df_almost_equals_pandas)
    apply_functions = [lambda df: -df, lambda df: df.sum(axis=1)]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.first())
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill())
    eval_prod(modin_groupby, pandas_groupby)
    eval_std(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.last())
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.cov(), modin_df_almost_equals_pandas)
    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)
    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.corr(), modin_df_almost_equals_pandas)
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_size(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take([0]))