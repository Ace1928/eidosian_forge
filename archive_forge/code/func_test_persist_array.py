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
@pytest.mark.skipif('not da')
def test_persist_array():
    from dask.array.utils import assert_eq
    arr = np.arange(100).reshape((10, 10))
    x = da.from_array(arr, chunks=(5, 5))
    x = x + 1 - x.mean(axis=0)
    y = x.persist()
    assert_eq(x, y)
    assert set(y.dask).issubset(x.dask)
    assert len(y.dask) == y.npartitions