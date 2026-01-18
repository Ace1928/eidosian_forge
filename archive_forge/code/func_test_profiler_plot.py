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
def test_profiler_plot():
    with prof:
        get(dsk, 'e')
    p = prof.visualize(width=500, height=300, tools='hover', title='Not the default', show=False, save=False)
    if BOKEH_VERSION().major < 3:
        assert p.plot_width == 500
        assert p.plot_height == 300
    else:
        assert p.width == 500
        assert p.height == 300
    assert len(p.tools) == 1
    assert isinstance(p.tools[0], bokeh.models.HoverTool)
    assert p.title.text == 'Not the default'
    prof.clear()
    with warnings.catch_warnings(record=True) as record:
        prof.visualize(show=False, save=False)
    assert not record