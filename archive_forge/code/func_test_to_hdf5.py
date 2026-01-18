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
def test_to_hdf5():
    h5py = pytest.importorskip('h5py')
    x = da.ones((4, 4), chunks=(2, 2))
    y = da.ones(4, chunks=2, dtype='i4')
    with tmpfile('.hdf5') as fn:
        x.to_hdf5(fn, '/x')
        with h5py.File(fn, mode='r+') as f:
            d = f['/x']
            assert_eq(d[:], x)
            assert d.chunks == (2, 2)
    with tmpfile('.hdf5') as fn:
        x.to_hdf5(fn, '/x', chunks=None)
        with h5py.File(fn, mode='r+') as f:
            d = f['/x']
            assert_eq(d[:], x)
            assert d.chunks is None
    with tmpfile('.hdf5') as fn:
        x.to_hdf5(fn, '/x', chunks=(1, 1))
        with h5py.File(fn, mode='r+') as f:
            d = f['/x']
            assert_eq(d[:], x)
            assert d.chunks == (1, 1)
    with tmpfile('.hdf5') as fn:
        da.to_hdf5(fn, {'/x': x, '/y': y})
        with h5py.File(fn, mode='r+') as f:
            assert_eq(f['/x'][:], x)
            assert f['/x'].chunks == (2, 2)
            assert_eq(f['/y'][:], y)
            assert f['/y'].chunks == (2,)