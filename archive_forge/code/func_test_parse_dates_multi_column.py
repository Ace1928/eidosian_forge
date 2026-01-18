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
def test_parse_dates_multi_column():
    pdmc_text = normalize_text('\n    ID,date,time\n    10,2003-11-04,180036\n    11,2003-11-05,125640\n    12,2003-11-01,2519\n    13,2003-10-22,142559\n    14,2003-10-24,163113\n    15,2003-10-20,170133\n    16,2003-11-11,160448\n    17,2003-11-03,171759\n    18,2003-11-07,190928\n    19,2003-10-21,84623\n    20,2003-10-25,192207\n    21,2003-11-13,180156\n    22,2003-11-15,131037\n    ')
    if PANDAS_GE_220:
        with pytest.warns(FutureWarning, match='nested'):
            with filetext(pdmc_text) as fn:
                ddf = dd.read_csv(fn, parse_dates=[['date', 'time']])
                df = pd.read_csv(fn, parse_dates=[['date', 'time']])
                assert (df.columns == ddf.columns).all()
                assert len(df) == len(ddf)
    else:
        with filetext(pdmc_text) as fn:
            ddf = dd.read_csv(fn, parse_dates=[['date', 'time']])
            df = pd.read_csv(fn, parse_dates=[['date', 'time']])
            assert (df.columns == ddf.columns).all()
            assert len(df) == len(ddf)