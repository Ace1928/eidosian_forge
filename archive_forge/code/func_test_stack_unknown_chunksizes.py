from __future__ import annotations
import contextlib
import copy
import pathlib
import re
import xml.etree.ElementTree
from unittest import mock
import pytest
import math
import operator
import os
import time
import warnings
from functools import reduce
from io import StringIO
from operator import add, sub
from threading import Lock
from tlz import concat, merge
from tlz.curried import identity
import dask
import dask.array as da
from dask.array.chunk import getitem
from dask.array.core import (
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reshape import _not_implemented_message
from dask.array.tests.test_dispatch import EncapsulateNDArray
from dask.array.utils import assert_eq, same_keys
from dask.base import compute_as_if_collection, tokenize
from dask.blockwise import broadcast_dimensions
from dask.blockwise import make_blockwise_graph as top
from dask.blockwise import optimize_blockwise
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import Blockwise
from dask.utils import SerializableLock, key_split, parse_bytes, tmpdir, tmpfile
from dask.utils_test import dec, hlg_layer_topological, inc
def test_stack_unknown_chunksizes():
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    a_df = pd.DataFrame({'x': np.arange(12)})
    b_df = pd.DataFrame({'y': np.arange(12) * 10})
    a_ddf = dd.from_pandas(a_df, sort=False, npartitions=3)
    b_ddf = dd.from_pandas(b_df, sort=False, npartitions=3)
    a_x = a_ddf.values
    b_x = b_ddf.values
    assert np.isnan(a_x.shape[0])
    assert np.isnan(b_x.shape[0])
    with pytest.raises(ValueError) as exc_info:
        da.stack([a_x, b_x], axis=0)
    assert 'shape' in str(exc_info.value)
    assert 'nan' in str(exc_info.value)
    c_x = da.stack([a_x, b_x], axis=0, allow_unknown_chunksizes=True)
    assert_eq(c_x, np.stack([a_df.values, b_df.values], axis=0))
    with pytest.raises(ValueError) as exc_info:
        da.stack([a_x, b_x], axis=1)
    assert 'shape' in str(exc_info.value)
    assert 'nan' in str(exc_info.value)
    c_x = da.stack([a_x, b_x], axis=1, allow_unknown_chunksizes=True)
    assert_eq(c_x, np.stack([a_df.values, b_df.values], axis=1))
    m_df = pd.DataFrame({'m': np.arange(12) * 100})
    n_df = pd.DataFrame({'n': np.arange(12) * 1000})
    m_ddf = dd.from_pandas(m_df, sort=False, npartitions=3)
    n_ddf = dd.from_pandas(n_df, sort=False, npartitions=3)
    m_x = m_ddf.values
    n_x = n_ddf.values
    assert np.isnan(m_x.shape[0])
    assert np.isnan(n_x.shape[0])
    with pytest.raises(ValueError) as exc_info:
        da.stack([[a_x, b_x], [m_x, n_x]])
    assert 'shape' in str(exc_info.value)
    assert 'nan' in str(exc_info.value)
    c_x = da.stack([[a_x, b_x], [m_x, n_x]], allow_unknown_chunksizes=True)
    assert_eq(c_x, np.stack([[a_df.values, b_df.values], [m_df.values, n_df.values]]))