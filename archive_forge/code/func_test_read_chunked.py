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
@pytest.mark.parametrize('block', [5, 15, 33, 200, 90000])
def test_read_chunked(block):
    with tmpdir() as path:
        fn = os.path.join(path, '1.json')
        df.to_json(fn, orient='records', lines=True)
        d = dd.read_json(fn, blocksize=block, sample=10)
        assert d.npartitions > 1 or block > 30
        assert_eq(d, df, check_index=False)