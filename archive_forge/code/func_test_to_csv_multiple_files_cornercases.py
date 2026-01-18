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
def test_to_csv_multiple_files_cornercases():
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        with pytest.raises(ValueError):
            fn = os.path.join(dn, 'data_*_*.csv')
            a.to_csv(fn)
    df16 = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        a.to_csv(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, mode='w', index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df)
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        a.to_csv(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, mode='w', index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)