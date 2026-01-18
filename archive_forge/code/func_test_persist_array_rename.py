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
def test_persist_array_rename():
    a = da.zeros(4, dtype=int, chunks=2)
    rebuild, args = a.__dask_postpersist__()
    dsk = {('b', 0): np.array([1, 2]), ('b', 1): np.array([3, 4])}
    b = rebuild(dsk, *args, rename={a.name: 'b'})
    assert isinstance(b, da.Array)
    assert b.name == 'b'
    assert b.__dask_keys__() == [('b', 0), ('b', 1)]
    da.utils.assert_eq(b, [1, 2, 3, 4])