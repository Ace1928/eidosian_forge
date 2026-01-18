from __future__ import annotations
import glob
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('index', [None, 'i32'])
@pytest.mark.parametrize('columns', [None, ['i32', 'i64', 'f']])
def test_orc_roundtrip(tmpdir, index, columns):
    tmp = str(tmpdir)
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'i64': np.arange(1000, dtype=np.int64), 'f': np.arange(1000, dtype=np.float64), 'bhello': np.random.choice(['hello', 'yo', 'people'], size=1000).astype('O')})
    if index:
        data = data.set_index(index)
    df = dd.from_pandas(data, chunksize=500)
    if columns:
        data = data[[c for c in columns if c != index]]
    df.to_orc(tmp, write_index=bool(index))
    df2 = dd.read_orc(tmp, index=index, columns=columns)
    assert_eq(data, df2, check_index=bool(index))