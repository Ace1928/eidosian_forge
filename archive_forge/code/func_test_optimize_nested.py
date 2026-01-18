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
def test_optimize_nested():
    a = dask.delayed(inc)(1)
    b = dask.delayed(inc)(a)
    c = a + b
    result = optimize({'a': a, 'b': [1, 2, b]}, (c, 2))
    a2 = result[0]['a']
    b2 = result[0]['b'][2]
    c2 = result[1][0]
    assert isinstance(a2, Delayed)
    assert isinstance(b2, Delayed)
    assert isinstance(c2, Delayed)
    assert dict(a2.dask) == dict(b2.dask) == dict(c2.dask)
    assert compute(*result) == ({'a': 2, 'b': [1, 2, 3]}, (5, 2))
    res = optimize([a, b], c, traverse=False)
    assert res[0][0] is a
    assert res[0][1] is b
    assert res[1].compute() == 5