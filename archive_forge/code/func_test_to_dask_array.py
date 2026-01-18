from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('lengths,as_frame,meta', [([2, 2, 1], False, None), (True, False, None), (True, False, np.array([], dtype='f4'))])
def test_to_dask_array(meta, as_frame, lengths):
    s = pd.Series([1, 2, 3, 4, 5], name='foo', dtype='i4')
    a = dd.from_pandas(s, chunksize=2)
    if as_frame:
        a = a.to_frame()
    result = a.to_dask_array(lengths=lengths, meta=meta)
    assert isinstance(result, da.Array)
    expected_chunks = ((2, 2, 1),)
    if as_frame:
        expected_chunks = expected_chunks + ((1,),)
    assert result.chunks == expected_chunks