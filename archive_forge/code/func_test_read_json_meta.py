from __future__ import annotations
import json
import os
import fsspec
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200
from dask.dataframe.utils import assert_eq
from dask.utils import tmpdir, tmpfile
@pytest.mark.parametrize('orient', ['split', 'records', 'index', 'columns', 'values'])
def test_read_json_meta(orient, tmpdir):
    df = pd.DataFrame({'x': range(5), 'y': ['a', 'b', 'c', 'd', 'e']})
    df2 = df.assign(x=df.x + 0.5)
    lines = orient == 'records'
    df.to_json(str(tmpdir.join('fil1.json')), orient=orient, lines=lines)
    df2.to_json(str(tmpdir.join('fil2.json')), orient=orient, lines=lines)
    sol = pd.concat([df, df2])
    meta = df2.iloc[:0]
    if orient == 'values':
        sol.columns = meta.columns = [0, 1]
    res = dd.read_json(str(tmpdir.join('fil*.json')), orient=orient, meta=meta, lines=lines)
    assert_eq(res, sol)
    if orient == 'records':
        res = dd.read_json(str(tmpdir.join('fil*.json')), orient=orient, meta=meta, lines=True, blocksize=50)
        assert_eq(res, sol, check_index=False)