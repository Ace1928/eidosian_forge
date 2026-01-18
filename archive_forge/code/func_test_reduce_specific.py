import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('fn', ['max', 'min', 'median', 'mean', 'skew', 'kurt', 'sem', 'std', 'var'])
@pytest.mark.parametrize('axis', [0, 1, None])
@pytest.mark.parametrize('numeric_only', [False, True])
def test_reduce_specific(fn, numeric_only, axis):
    expected_exception = None
    if not numeric_only:
        if fn in ('max', 'min'):
            if axis == 0:
                operator = '>=' if fn == 'max' else '<='
                expected_exception = TypeError(f"'{operator}' not supported between instances of 'str' and 'float'")
                if StorageFormat.get() == 'Hdk':
                    expected_exception = False
            else:
                expected_exception = False
        elif fn in ('skew', 'kurt', 'sem', 'std', 'var', 'median', 'mean'):
            expected_exception = False
    eval_general(*create_test_dfs(test_data_diff_dtype), lambda df: getattr(df, fn)(numeric_only=numeric_only, axis=axis), expected_exception=expected_exception)