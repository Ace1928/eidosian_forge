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
@pytest.mark.skipif('not psutil')
def test_resource_profiler_plot():
    with ResourceProfiler(dt=0.01) as rprof:
        get(dsk2, 'c')
    p = rprof.visualize(width=500, height=300, tools='hover', title='Not the default', show=False, save=False)
    if BOKEH_VERSION().major < 3:
        assert p.plot_width == 500
        assert p.plot_height == 300
    else:
        assert p.width == 500
        assert p.height == 300
    assert len(p.tools) == 1
    assert isinstance(p.tools[0], bokeh.models.HoverTool)
    assert p.title.text == 'Not the default'
    rprof.clear()
    for results in [[], [(1.0, 0, 0)]]:
        rprof.results = results
        rprof.start_time = 0.0
        rprof.end_time = 1.0
        with warnings.catch_warnings(record=True) as record:
            p = rprof.visualize(show=False, save=False)
        assert not record
        assert p.x_range.start == 0
        assert p.x_range.end == 1
        assert p.y_range.start == 0
        assert p.y_range.end == 100
        assert p.extra_y_ranges['memory'].start == 0
        assert p.extra_y_ranges['memory'].end == 100