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
def test_persist_item_change_name():
    a = db.from_sequence([1, 2, 3]).min()
    rebuild, args = a.__dask_postpersist__()
    b = rebuild({'x': 4}, *args, rename={a.name: 'x'})
    assert isinstance(b, db.Item)
    assert b.__dask_keys__() == ['x']
    db.utils.assert_eq(b, 4)