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
def test_windows_line_terminator():
    text = 'a,b\r\n1,2\r\n2,3\r\n3,4\r\n4,5\r\n5,6\r\n6,7'
    with filetext(text) as fn:
        df = dd.read_csv(fn, blocksize=5, lineterminator='\r\n')
        assert df.b.sum().compute() == 2 + 3 + 4 + 5 + 6 + 7
        assert df.a.sum().compute() == 1 + 2 + 3 + 4 + 5 + 6