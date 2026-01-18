from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
@pytest.mark.xfail(reason='blockwise fusion does not respect this, which is ok')
def test_turn_off_fusion():
    x = da.ones(10, chunks=(5,))
    y = da.sum(x + 1 + 2 + 3)
    a = y.__dask_optimize__(y.dask, y.__dask_keys__())
    with dask.config.set({'optimization.fuse.ave-width': 0}):
        b = y.__dask_optimize__(y.dask, y.__dask_keys__())
    assert dask.get(a, y.__dask_keys__()) == dask.get(b, y.__dask_keys__())
    assert len(a) < len(b)