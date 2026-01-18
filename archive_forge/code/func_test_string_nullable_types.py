from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_string_nullable_types(df_ddf):
    df, ddf = df_ddf
    assert_eq(ddf.string_col.str.count('A'), df.string_col.str.count('A'))
    assert_eq(ddf.string_col.str.isalpha(), df.string_col.str.isalpha())