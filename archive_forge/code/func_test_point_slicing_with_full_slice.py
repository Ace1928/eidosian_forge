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
def test_point_slicing_with_full_slice():
    from dask.array.core import _get_axis, _vindex_transpose
    x = np.arange(4 * 5 * 6 * 7).reshape((4, 5, 6, 7))
    d = da.from_array(x, chunks=(2, 3, 3, 4))
    inds = [[[1, 2, 3], None, [3, 2, 1], [5, 3, 4]], [[1, 2, 3], None, [4, 3, 2], None], [[1, 2, 3], [3, 2, 1]], [[1, 2, 3], [3, 2, 1], [3, 2, 1], [5, 3, 4]], [[], [], [], None], [np.array([1, 2, 3]), None, np.array([4, 3, 2]), None], [None, None, [1, 2, 3], [4, 3, 2]], [None, [0, 2, 3], None, [0, 3, 2]]]
    for ind in inds:
        slc = [i if isinstance(i, (np.ndarray, list)) else slice(None, None) for i in ind]
        result = d.vindex[tuple(slc)]
        axis = _get_axis(ind)
        expected = _vindex_transpose(x[tuple(slc)], axis)
        assert_eq(result, expected)
        k = len(next((i for i in ind if isinstance(i, (np.ndarray, list)))))
        assert result.shape[0] == k