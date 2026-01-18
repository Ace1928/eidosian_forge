from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_series_format():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list('ABCDEFGH'))
    ds = dd.from_pandas(s, 3)
    footer = _format_footer()
    exp = dedent(f'    Dask Series Structure:\n    npartitions=3\n    A    int64\n    D      ...\n    G      ...\n    H      ...\n    dtype: int64\n    {footer}')
    assert repr(ds) == exp
    assert str(ds) == exp
    exp = dedent('    npartitions=3\n    A    int64\n    D      ...\n    G      ...\n    H      ...')
    assert ds.to_string() == exp
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list('ABCDEFGH'), name='XXX')
    ds = dd.from_pandas(s, 3)
    exp = dedent(f'    Dask Series Structure:\n    npartitions=3\n    A    int64\n    D      ...\n    G      ...\n    H      ...\n    Name: XXX, dtype: int64\n    {footer}')
    assert repr(ds) == exp
    assert str(ds) == exp