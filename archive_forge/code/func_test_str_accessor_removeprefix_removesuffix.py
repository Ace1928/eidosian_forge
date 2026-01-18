from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.mark.skipif(not PANDAS_GE_140, reason='requires pandas >= 1.4.0')
@pytest.mark.parametrize('method', ['removeprefix', 'removesuffix'])
def test_str_accessor_removeprefix_removesuffix(df_ddf, method):
    df, ddf = df_ddf
    prefix = df.str_col.iloc[0][:2]
    suffix = df.str_col.iloc[0][-2:]
    missing = 'definitely a missing prefix/suffix'

    def call(df, arg):
        return getattr(df.str_col.str, method)(arg)
    assert_eq(call(ddf, prefix), call(df, prefix))
    assert_eq(call(ddf, suffix), call(df, suffix))
    assert_eq(call(ddf, missing), call(df, missing))