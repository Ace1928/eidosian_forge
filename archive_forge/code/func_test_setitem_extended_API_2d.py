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
@pytest.mark.parametrize('index, value', [[Ellipsis, -1], [(slice(None, None, 2), slice(None, None, -1)), -1], [slice(1, None, 2), -1], [[4, 3, 1], -1], [(Ellipsis, 4), -1], [5, -1], [(slice(None), 2), range(6)], [3, range(10)], [(slice(None), [3, 5, 6]), [-30, -31, -32]], [([-1, 0, 1], 2), [-30, -31, -32]], [(slice(None, 2), slice(None, 3)), [-50, -51, -52]], [(slice(None), [6, 1, 3]), [-60, -61, -62]], [(slice(1, 3), slice(1, 4)), [[-70, -71, -72]]], [(slice(None), [9, 8, 8]), [-80, -81, 91]], [([True, False, False, False, True, False], 2), -1], [(3, [True, True, False, True, True, False, True, False, True, True]), -1], [(np.array([False, False, True, True, False, False]), slice(5, 7)), -1], [(4, da.from_array([False, False, True, True, False, False, True, False, False, True])), -1], [(slice(2, 4), da.from_array([False, False, True, True, False, False, True, False, False, True])), [[-100, -101, -102, -103], [-200, -201, -202, -203]]], [slice(5, None, 2), -99], [slice(5, None, 2), range(1, 11)], [slice(1, None, -2), -98], [slice(1, None, -2), range(11, 21)]])
def test_setitem_extended_API_2d(index, value):
    x = np.ma.arange(60).reshape((6, 10))
    dx = da.from_array(x, chunks=(2, 3))
    dx[index] = value
    x[index] = value
    assert_eq(x, dx.compute())