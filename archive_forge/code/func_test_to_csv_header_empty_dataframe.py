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
@pytest.mark.parametrize('header, expected', [(False, ''), (True, 'x,y\n')])
def test_to_csv_header_empty_dataframe(header, expected):
    dfe = pd.DataFrame({'x': [], 'y': []})
    ddfe = dd.from_pandas(dfe, npartitions=1)
    with tmpdir() as dn:
        ddfe.to_csv(os.path.join(dn, 'fooe*.csv'), index=False, header=header)
        assert not os.path.exists(os.path.join(dn, 'fooe1.csv'))
        filename = os.path.join(dn, 'fooe0.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected
        os.remove(filename)