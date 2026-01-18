import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('retbins', bool_arg_values, ids=bool_arg_keys)
def test_qcut(retbins):
    pandas_series = pandas.Series(range(10))
    modin_series = pd.Series(range(10))
    pandas_result = pandas.qcut(pandas_series, 4, retbins=retbins)
    with warns_that_defaulting_to_pandas():
        modin_result = pd.qcut(modin_series, 4, retbins=retbins)
    if retbins:
        df_equals(modin_result[0], pandas_result[0])
        df_equals(modin_result[0].cat.categories, pandas_result[0].cat.categories)
        assert_array_equal(modin_result[1], pandas_result[1])
    else:
        df_equals(modin_result, pandas_result)
        df_equals(modin_result.cat.categories, pandas_result.cat.categories)
    pandas_result = pandas.qcut(range(5), 4)
    modin_result = pd.qcut(range(5), 4)
    df_equals(modin_result, pandas_result)