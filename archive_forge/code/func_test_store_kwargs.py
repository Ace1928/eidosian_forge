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
def test_store_kwargs():
    d = da.ones((10, 10), chunks=(2, 2))
    a = d + 1
    called = [False]

    def get_func(*args, **kwargs):
        assert kwargs.pop('foo') == 'test kwarg'
        r = dask.get(*args, **kwargs)
        called[0] = True
        return r
    called[0] = False
    at = np.zeros(shape=(10, 10))
    store([a], [at], scheduler=get_func, foo='test kwarg')
    assert called[0]
    called[0] = False
    at = np.zeros(shape=(10, 10))
    a.store(at, scheduler=get_func, foo='test kwarg')
    assert called[0]
    called[0] = False
    at = np.zeros(shape=(10, 10))
    store([a], [at], scheduler=get_func, return_stored=True, foo='test kwarg')
    assert called[0]