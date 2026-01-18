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
@pytest.mark.skipif('not bokeh')
def test_saves_file():
    with tmpfile('html') as fn:
        with prof:
            get(dsk, 'e')
        prof.visualize(show=False, filename=fn)
        assert os.path.exists(fn)
        with open(fn) as f:
            assert 'html' in f.read().lower()