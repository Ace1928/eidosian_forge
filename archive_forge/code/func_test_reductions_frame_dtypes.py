from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.parametrize('func, kwargs', [('sum', None), ('prod', None), ('product', None), ('mean', None), ('std', None), ('std', {'ddof': 0}), ('std', {'skipna': False}), ('std', {'ddof': 0, 'skipna': False}), ('min', None), ('max', None), ('count', None), ('sem', None), ('sem', {'ddof': 0}), ('sem', {'skipna': False}), ('sem', {'ddof': 0, 'skipna': False}), ('var', None), ('var', {'ddof': 0}), ('var', {'skipna': False}), ('var', {'ddof': 0, 'skipna': False})])
@pytest.mark.parametrize('numeric_only', [None, True, pytest.param(False, marks=pytest.mark.xfail(True, reason='numeric_only=False not implemented', strict=False))])
def test_reductions_frame_dtypes(func, kwargs, numeric_only):
    if pyarrow_strings_enabled() and func == 'sum' and (numeric_only is None):
        pytest.xfail('Known failure with pyarrow strings')
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2011, i, 1) for i in range(1, 8)], 'str': list('abcdefgh'), 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, np.nan]), 'bool': [True, False] * 4})
    if kwargs is None:
        kwargs = {}
    if numeric_only is False or numeric_only is None:
        if func in ('sum', 'prod', 'product', 'mean', 'median', 'std', 'sem', 'var'):
            df = df.drop(columns=['dt', 'timedelta'])
        if func in ('prod', 'product', 'mean', 'std', 'sem', 'var'):
            df = df.drop(columns=['str'])
    if numeric_only is not None:
        kwargs['numeric_only'] = numeric_only
    ddf = dd.from_pandas(df, 3)
    with check_numeric_only_deprecation():
        expected = getattr(df, func)(**kwargs)
        actual = getattr(ddf, func)(**kwargs)
        assert_eq(expected, actual)