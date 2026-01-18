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
@pytest.mark.skipif('not dd')
def test_persist_scalar_rename():
    import dask.dataframe as dd
    if dd._dask_expr_enabled():
        pytest.skip("doesn't make sense")
    ds1 = pd.Series([1, 2, 3, 4])
    dds1 = dd.from_pandas(ds1, npartitions=2).min()
    rebuild, args = dds1.__dask_postpersist__()
    dds2 = rebuild({('x', 0): 5}, *args, rename={dds1._name: 'x'})
    assert dds2.__dask_keys__() == [('x', 0)]
    dd.utils.assert_eq(dds2, 5)