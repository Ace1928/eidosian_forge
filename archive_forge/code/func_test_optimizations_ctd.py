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
def test_optimizations_ctd():
    da = pytest.importorskip('dask.array')
    x = da.arange(2, chunks=1)[:1]
    dsk1 = collections_to_dsk([x])
    with dask.config.set({'optimizations': [lambda dsk, keys: dsk]}):
        dsk2 = collections_to_dsk([x])
    assert dsk1 == dsk2