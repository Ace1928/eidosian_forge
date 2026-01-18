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
def test_optimize_blockwise_custom_annotations():
    a = da.ones(10, chunks=(5,))
    b = a + 1
    with dask.annotate(qux='foo'):
        c = b + 2
        d = c + 3
    with dask.annotate(qux='baz'):
        e = d + 4
        f = e + 5
    g = f + 6
    dsk = da.optimization.optimize_blockwise(g.dask)
    annotations = (layer.annotations for layer in dsk.layers.values() if isinstance(layer, Blockwise))
    annotations = collections.Counter((tuple(a.items()) if type(a) is dict else a for a in annotations))
    assert len(annotations) == 3
    assert annotations[None] == 2
    assert annotations[('qux', 'baz'),] == 1
    assert annotations[('qux', 'foo'),] == 1