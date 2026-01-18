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
def test_store_delayed_target():
    from dask.delayed import delayed
    d = da.ones((4, 4), chunks=(2, 2))
    a, b = (d + 1, d + 2)
    targs = {}

    def make_target(key):
        a = np.empty((4, 4))
        targs[key] = a
        return a
    atd = delayed(make_target)('at')
    btd = delayed(make_target)('bt')
    st = store([a, b], [atd, btd])
    at = targs['at']
    bt = targs['bt']
    assert st is None
    assert_eq(at, a)
    assert_eq(bt, b)
    for st_compute in [False, True]:
        targs.clear()
        st = store([a, b], [atd, btd], return_stored=True, compute=st_compute)
        if st_compute:
            assert all((not any(dask.core.get_deps(e.dask)[0].values()) for e in st))
        st = dask.compute(*st)
        at = targs['at']
        bt = targs['bt']
        assert st is not None
        assert isinstance(st, tuple)
        assert all([isinstance(v, np.ndarray) for v in st])
        assert_eq(at, a)
        assert_eq(bt, b)
        assert_eq(st[0], a)
        assert_eq(st[1], b)
        pytest.raises(ValueError, lambda at=at, bt=bt: store([a], [at, bt]))
        pytest.raises(ValueError, lambda at=at: store(at, at))
        pytest.raises(ValueError, lambda at=at, bt=bt: store([at, bt], [at, bt]))