import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', [test_data['int_data']], ids=['int_data'])
@pytest.mark.parametrize('index', [pytest.param(lambda df: df.columns[0], id='single_index_col'), pytest.param(lambda df: [*df.columns[0:2], *df.columns[-7:-4]], id='multiple_index_cols'), pytest.param(None, id='default_index')])
@pytest.mark.parametrize('columns', [pytest.param(lambda df: df.columns[len(df.columns) // 2], id='single_col'), pytest.param(lambda df: [*df.columns[len(df.columns) // 2:len(df.columns) // 2 + 4], df.columns[-7]], id='multiple_cols'), pytest.param(None, id='default_columns')])
@pytest.mark.parametrize('values', [pytest.param(lambda df: df.columns[-1], id='single_value_col'), pytest.param(lambda df: df.columns[-4:-1], id='multiple_value_cols'), pytest.param(None, id='default_values')])
@pytest.mark.parametrize('aggfunc', [pytest.param(np.mean, id='callable_tree_reduce_func'), pytest.param('mean', id='tree_reduce_func'), pytest.param('nunique', id='full_axis_func')])
def test_pivot_table_data(data, index, columns, values, aggfunc, request):
    if 'callable_tree_reduce_func-single_value_col-multiple_cols-multiple_index_cols' in request.node.callspec.id or 'callable_tree_reduce_func-multiple_value_cols-multiple_cols-multiple_index_cols' in request.node.callspec.id or 'tree_reduce_func-single_value_col-multiple_cols-multiple_index_cols' in request.node.callspec.id or ('tree_reduce_func-multiple_value_cols-multiple_cols-multiple_index_cols' in request.node.callspec.id) or ('full_axis_func-single_value_col-multiple_cols-multiple_index_cols' in request.node.callspec.id) or ('full_axis_func-multiple_value_cols-multiple_cols-multiple_index_cols' in request.node.callspec.id):
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7011')
    md_df, pd_df = create_test_dfs(data)
    if values is None:
        md_df, pd_df = (md_df.iloc[:42, :42], pd_df.iloc[:42, :42])
    expected_exception = None
    if 'default_columns-default_index' in request.node.callspec.id:
        expected_exception = ValueError('No group keys passed!')
    elif 'callable_tree_reduce_func' in request.node.callspec.id and 'int_data' in request.node.callspec.id:
        expected_exception = TypeError("'numpy.float64' object is not callable")
    eval_general(md_df, pd_df, operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs).sort_index(axis=int(index is not None)), index=index, columns=columns, values=values, aggfunc=aggfunc, expected_exception=expected_exception)