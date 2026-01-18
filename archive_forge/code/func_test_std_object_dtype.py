from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('func', [pytest.param('var', marks=pytest.mark.xfail(PANDAS_GE_200, reason='numeric_only=False not implemented')), pytest.param('std', marks=pytest.mark.xfail(PANDAS_GE_200, reason='numeric_only=False not implemented')), pytest.param('mean', marks=pytest.mark.xfail(PANDAS_GE_200, reason='numeric_only=False not implemented')), pytest.param('sum', marks=pytest.mark.xfail(pyarrow_strings_enabled(), reason='works in dask-expr'))])
def test_std_object_dtype(func):
    df = pd.DataFrame({'x': [1, 2, 1], 'y': ['a', 'b', 'c'], 'z': [11.0, 22.0, 33.0]})
    ddf = dd.from_pandas(df, npartitions=2)
    ctx = contextlib.nullcontext()
    if func != 'sum':
        ctx = check_nuisance_columns_warning()
    with ctx, check_numeric_only_deprecation():
        expected = getattr(df, func)()
    with _check_warning(func in ['std', 'var'] and (not PANDAS_GE_200), FutureWarning, message='numeric_only'):
        result = getattr(ddf, func)()
    assert_eq(expected, result)
    with record_numeric_only_warnings() as rec_pd:
        expected = getattr(df.groupby('x'), func)()
    with record_numeric_only_warnings() as rec_dd:
        result = getattr(ddf.groupby('x'), func)()
    assert len(rec_pd) == len(rec_dd)
    assert_eq(expected, result)
    with record_numeric_only_warnings() as rec_pd:
        expected = getattr(df.groupby('x').z, func)()
    with record_numeric_only_warnings() as rec_dd:
        result = getattr(ddf.groupby('x').z, func)()
    assert len(rec_pd) == len(rec_dd)
    assert_eq(expected, result)