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
@pytest.mark.parametrize('columns', ['volume', 'date', ['volume'], ('volume',), pandas.Series(['volume']), pandas.Index(['volume']), ['volume', 'volume', 'volume'], ['volume', 'price', 'date']], ids=['column', 'only_missed_column', 'list', 'tuple', 'series', 'index', 'duplicate_column', 'missed_column'])
def test_resample_getitem(columns, request):
    index = pandas.date_range('1/1/2013', periods=9, freq='min')
    data = {'price': range(9), 'volume': range(10, 19)}
    expected_exception = None
    if 'only_missed_column' in request.node.callspec.id:
        expected_exception = KeyError('Column not found: date')
    elif 'missed_column' in request.node.callspec.id:
        expected_exception = KeyError("Columns not found: 'date'")
    eval_general(*create_test_dfs(data, index=index), lambda df: df.resample('3min')[columns].mean(), expected_exception=expected_exception)