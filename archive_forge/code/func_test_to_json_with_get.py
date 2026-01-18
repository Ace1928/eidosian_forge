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
def test_to_json_with_get():
    from dask.multiprocessing import get as mp_get
    flag = [False]

    def my_get(*args, **kwargs):
        flag[0] = True
        return mp_get(*args, **kwargs)
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpdir() as dn:
        ddf.to_json(dn, compute_kwargs={'scheduler': my_get})
        assert flag[0]
        result = dd.read_json(os.path.join(dn, '*'))
        assert_eq(result, df, check_index=False)