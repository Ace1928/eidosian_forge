from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
@pytest.mark.parametrize('npartitions', [1, 4])
def test_repartition_on_categoricals(npartitions):
    df = pd.DataFrame({'x': range(10), 'y': list('abababcbcb')})
    if pyarrow_strings_enabled():
        df = to_pyarrow_string(df)
    ddf = dd.from_pandas(df, npartitions=2)
    ddf['y'] = ddf['y'].astype('category')
    ddf2 = ddf.repartition(npartitions=npartitions)
    df = df.copy()
    df['y'] = df['y'].astype('category')
    assert_eq(df, ddf)
    assert_eq(df, ddf2)