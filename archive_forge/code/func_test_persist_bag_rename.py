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
def test_persist_bag_rename():
    a = db.from_sequence([1, 2, 3], npartitions=2)
    rebuild, args = a.__dask_postpersist__()
    dsk = {('b', 0): [4], ('b', 1): [5, 6]}
    b = rebuild(dsk, *args, rename={a.name: 'b'})
    assert isinstance(b, db.Bag)
    assert b.name == 'b'
    assert b.__dask_keys__() == [('b', 0), ('b', 1)]
    db.utils.assert_eq(b, [4, 5, 6])