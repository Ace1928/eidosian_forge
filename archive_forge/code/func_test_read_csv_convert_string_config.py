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
@pytest.mark.skipif(not PANDAS_GE_200, reason='dataframe.convert-string requires pandas>=2.0')
def test_read_csv_convert_string_config():
    pytest.importorskip('pyarrow', reason='Requires pyarrow strings')
    with filetext(csv_text) as fn:
        df = pd.read_csv(fn)
        with dask.config.set({'dataframe.convert-string': True}):
            ddf = dd.read_csv(fn)
        df_pyarrow = df.astype({'name': 'string[pyarrow]'})
        assert_eq(df_pyarrow, ddf, check_index=False)