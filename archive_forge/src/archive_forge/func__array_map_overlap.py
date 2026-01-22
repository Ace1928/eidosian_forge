from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def _array_map_overlap(tmpdir):
    da = pytest.importorskip('dask.array')
    array = da.ones((100,))
    return array.map_overlap(lambda x: x, depth=1, boundary='none')