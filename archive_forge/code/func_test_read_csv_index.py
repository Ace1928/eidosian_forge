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
def test_read_csv_index():
    with filetext(csv_text) as fn:
        f = dd.read_csv(fn, blocksize=20).set_index('amount')
        result = f.compute(scheduler='sync')
        assert result.index.name == 'amount'
        blocks = compute_as_if_collection(dd.DataFrame, f.dask, f.__dask_keys__(), scheduler='sync')
        for i, block in enumerate(blocks):
            if i < len(f.divisions) - 2:
                assert (block.index < f.divisions[i + 1]).all()
            if i > 0:
                assert (block.index >= f.divisions[i]).all()
        expected = pd.read_csv(fn).set_index('amount')
        assert_eq(result, expected)