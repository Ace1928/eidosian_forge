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
@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_read_csv_files(dd_read, pd_read, files):
    expected = read_files()
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv')
        assert_eq(df, expected, check_dtype=False)
        fn = '2014-01-01.csv'
        df = dd_read(fn)
        expected2 = pd_read(BytesIO(files[fn]))
        assert_eq(df, expected2, check_dtype=False)