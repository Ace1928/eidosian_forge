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
def test_read_csv_groupby_get_group(tmpdir):
    path = os.path.join(str(tmpdir), 'test.csv')
    df1 = pd.DataFrame([{'foo': 10, 'bar': 4}])
    df1.to_csv(path, index=False)
    ddf1 = dd.read_csv(path)
    ddfs = ddf1.groupby('foo')
    assert_eq(df1, ddfs.get_group(10).compute())