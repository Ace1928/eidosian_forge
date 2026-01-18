from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_series_format_long():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10, index=list('ABCDEFGHIJ') * 10)
    ds = dd.from_pandas(s, 10)
    footer = _format_footer()
    exp = dedent(f'        Dask Series Structure:\n        npartitions=10\n        A    int64\n        B      ...\n             ...  \n        J      ...\n        J      ...\n        dtype: int64\n        {footer}')
    assert repr(ds) == exp
    assert str(ds) == exp
    exp = dedent('    npartitions=10\n    A    int64\n    B      ...\n         ...  \n    J      ...\n    J      ...')
    assert ds.to_string() == exp