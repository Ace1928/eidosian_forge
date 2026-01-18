from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_str_accessor_cat(df_ddf):
    df, ddf = df_ddf
    sol = df.str_col.str.cat(df.str_col.str.upper(), sep=':')
    assert_eq(ddf.str_col.str.cat(ddf.str_col.str.upper(), sep=':'), sol)
    assert_eq(ddf.str_col.str.cat(df.str_col.str.upper(), sep=':'), sol)
    assert_eq(ddf.str_col.str.cat([ddf.str_col.str.upper(), df.str_col.str.lower()], sep=':'), df.str_col.str.cat([df.str_col.str.upper(), df.str_col.str.lower()], sep=':'))
    assert_eq(ddf.str_col.str.cat(sep=':'), df.str_col.str.cat(sep=':'))
    for o in ['foo', ['foo']]:
        with pytest.raises(TypeError):
            ddf.str_col.str.cat(o)