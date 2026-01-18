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
@pytest.mark.parametrize('rule', ['5min'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
@pytest.mark.parametrize('method_arg', [('pipe', lambda x: x.max() - x.min()), ('transform', lambda x: (x - x.mean()) / x.std()), ('apply', ['sum', 'mean', 'max']), ('aggregate', ['sum', 'mean', 'max'])])
def test_resampler_functions_with_arg(rule, axis, method_arg):
    data, index = (test_data_resample['data'], test_data_resample['index'])
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    if axis == 'columns':
        columns = pandas.date_range('31/12/2000', periods=len(pandas_df.columns), freq='min')
        modin_df.columns = columns
        pandas_df.columns = columns
    method, arg = (method_arg[0], method_arg[1])
    expected_exception = None
    if method in ('apply', 'aggregate'):
        expected_exception = NotImplementedError('axis other than 0 is not supported')
    eval_general(modin_df, pandas_df, lambda df: getattr(df.resample(rule, axis=axis), method)(arg), expected_exception=expected_exception)