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
def test_zarr_regions():
    zarr = pytest.importorskip('zarr')
    a = da.arange(16).reshape((4, 4)).rechunk(2)
    z = zarr.zeros_like(a, chunks=2)
    a[:2, :2].to_zarr(z, region=(slice(2), slice(2)))
    a2 = da.from_zarr(z)
    expected = [[0, 1, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert_eq(a2, expected)
    assert a2.chunks == a.chunks
    a[:3, 3:4].to_zarr(z, region=(slice(1, 4), slice(2, 3)))
    a2 = da.from_zarr(z)
    expected = [[0, 1, 0, 0], [4, 5, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0]]
    assert_eq(a2, expected)
    assert a2.chunks == a.chunks
    a[3:, 3:].to_zarr(z, region=(slice(2, 3), slice(1, 2)))
    a2 = da.from_zarr(z)
    expected = [[0, 1, 0, 0], [4, 5, 3, 0], [0, 15, 7, 0], [0, 0, 11, 0]]
    assert_eq(a2, expected)
    assert a2.chunks == a.chunks
    with pytest.raises(ValueError):
        with tmpdir() as d:
            a.to_zarr(d, region=(slice(2), slice(2)))