from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
def test_optimize_blockwise_control_annotations():
    """
    Can we fuse blockwise layers with different, but compatible
    annotations for retries, priority, etc.
    """
    a = da.ones(10, chunks=(5,))
    b = a + 1
    with dask.annotate(retries=5, workers=['a', 'b', 'c'], allow_other_workers=False):
        c = b + 2
    with dask.annotate(priority=2, workers=['b', 'c', 'd'], allow_other_workers=True):
        d = c + 3
    with dask.annotate(retries=3, resources={'GPU': 2, 'Memory': 10}):
        e = d + 4
    with dask.annotate(priority=4, resources={'GPU': 5, 'Memory': 4}):
        f = e + 5
    with dask.annotate(foo='bar'):
        g = f + 6
    h = g + 6
    dsk = da.optimization.optimize_blockwise(h.dask)
    assert len(dsk.layers) == 3
    layer = hlg_layer_topological(dsk, 0)
    annotations = layer.annotations
    assert len(annotations) == 5
    assert annotations['priority'] == 4
    assert annotations['retries'] == 5
    assert annotations['allow_other_workers'] is False
    assert set(annotations['workers']) == {'b', 'c'}
    assert annotations['resources'] == {'GPU': 5, 'Memory': 10}
    with dask.config.set({'optimization.annotations.fuse': False}):
        dsk = da.optimization.optimize_blockwise(h.dask)
        assert len(dsk.layers) == 7