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
@pytest.mark.parametrize('limit', [None, 134217728, 67108864])
@pytest.mark.parametrize('shape, chunks, reshape_size', [((300, 180, 4, 18483), (-1, -1, 1, 183), (300, 180, -1)), ((300, 300, 4, 18483), (-1, -1, 1, 183), (300, 300, -1))])
def test_reshape_avoids_large_chunks(limit, shape, chunks, reshape_size):
    array = da.random.default_rng().random(shape, chunks=chunks)
    if limit is None:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            result = array.reshape(*reshape_size, limit=limit)
    else:
        result = array.reshape(*reshape_size, limit=limit)
    nbytes = array.dtype.itemsize
    max_chunksize_in_bytes = reduce(operator.mul, result.chunksize) * nbytes
    if limit is None:
        limit = parse_bytes(dask.config.get('array.chunk-size'))
    assert max_chunksize_in_bytes < limit