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
def test_persist_series_rename():
    import dask.dataframe as dd
    if dd._dask_expr_enabled():
        pytest.skip("doesn't make sense")
    ds1 = pd.Series([1, 2, 3, 4])
    ds2 = pd.Series([5, 6, 7, 8])
    dds1 = dd.from_pandas(ds1, npartitions=2)
    rebuild, args = dds1.__dask_postpersist__()
    dsk = {('x', 0): ds2.iloc[:2], ('x', 1): ds2.iloc[2:]}
    dds2 = rebuild(dsk, *args, rename={dds1._name: 'x'})
    assert dds2.__dask_keys__() == [('x', 0), ('x', 1)]
    dd.utils.assert_eq(dds2, ds2)