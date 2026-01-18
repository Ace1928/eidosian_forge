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
@pytest.mark.parametrize('op, make_args', [('align', lambda df: {'other': df}), ('corrwith', lambda df: {'other': df}), ('ewm', lambda df: {'com': 0.5}), ('from_dict', lambda df: {'data': None}), ('from_records', lambda df: {'data': to_pandas(df)}), ('hist', lambda df: {'column': 'int_col'}), ('interpolate', None), ('mask', lambda df: {'cond': df != 0}), ('pct_change', None), ('to_xarray', None), ('flags', None), ('set_flags', lambda df: {'allows_duplicate_labels': False})])
def test_ops_defaulting_to_pandas(op, make_args):
    modin_df = pd.DataFrame(test_data_diff_dtype).drop(['str_col', 'bool_col'], axis=1)
    with warns_that_defaulting_to_pandas():
        operation = getattr(modin_df, op)
        if make_args is not None:
            operation(**make_args(modin_df))
        else:
            try:
                operation()
            except TypeError:
                pass