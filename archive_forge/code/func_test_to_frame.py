from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
@pytest.mark.parametrize('index', [pd.date_range('2011-01-01', freq='h', periods=100), range(100)])
def test_to_frame(index):
    df = pd.DataFrame({'A': np.random.randn(100)}, index=index)
    ddf = dd.from_pandas(df, 10)
    expected = df.index.to_frame()
    actual = ddf.index.to_frame()
    assert actual.known_divisions
    assert_eq(expected, actual)
    assert_eq(df.index.to_frame(name='foo'), ddf.index.to_frame(name='foo'))