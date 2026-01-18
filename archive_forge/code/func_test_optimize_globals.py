from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
def test_optimize_globals():
    da = pytest.importorskip('dask.array')
    x = da.ones(10, chunks=(5,))

    def optimize_double(dsk, keys):
        return {k: (mul, 2, v) for k, v in dsk.items()}
    from dask.array.utils import assert_eq
    assert_eq(x + 1, np.ones(10) + 1)
    with dask.config.set(array_optimize=optimize_double):
        assert_eq(x + 1, (np.ones(10) * 2 + 1) * 2, check_chunks=False)
    assert_eq(x + 1, np.ones(10) + 1)
    b = db.range(10, npartitions=2)
    with dask.config.set(array_optimize=optimize_double):
        xx, bb = dask.compute(x + 1, b.map(inc), scheduler='single-threaded')
        assert_eq(xx, (np.ones(10) * 2 + 1) * 2)