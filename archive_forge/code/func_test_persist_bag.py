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
def test_persist_bag():
    a = db.from_sequence([1, 2, 3], npartitions=2).map(lambda x: x * 2)
    assert len(a.__dask_graph__()) == 4
    b = a.persist(scheduler='sync')
    assert isinstance(b, db.Bag)
    assert len(b.__dask_graph__()) == 2
    db.utils.assert_eq(a, b)