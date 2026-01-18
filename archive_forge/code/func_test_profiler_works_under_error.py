from __future__ import annotations
import contextlib
import os
import warnings
from operator import add, mul
from timeit import default_timer
import pytest
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler
from dask.diagnostics.profile_visualize import BOKEH_VERSION
from dask.threaded import get
from dask.utils import apply, tmpfile
from dask.utils_test import slowadd
def test_profiler_works_under_error():
    div = lambda x, y: x / y
    dsk = {'x': (div, 1, 1), 'y': (div, 'x', 2), 'z': (div, 'y', 0)}
    with contextlib.suppress(ZeroDivisionError):
        with prof:
            get(dsk, 'z')
    assert all((len(v) == 5 for v in prof.results))
    assert len(prof.results) == 2