from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
def test_to_csv_with_get():
    from dask.multiprocessing import get as mp_get
    flag = [False]

    def my_get(*args, **kwargs):
        flag[0] = True
        return mp_get(*args, **kwargs)
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpdir() as dn:
        ddf.to_csv(dn, index=False, compute_kwargs={'scheduler': my_get})
        assert flag[0]
        result = dd.read_csv(os.path.join(dn, '*'))
        assert_eq(result, df, check_index=False)