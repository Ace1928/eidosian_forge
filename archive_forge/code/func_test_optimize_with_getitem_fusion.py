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
def test_optimize_with_getitem_fusion(getter):
    dsk = {'a': 'some-array', 'b': (getter, 'a', (slice(10, 20), slice(100, 200))), 'c': (getter, 'b', (5, slice(50, 60)))}
    result = optimize(dsk, ['c'])
    expected_task = (getter, 'some-array', (15, slice(150, 160)))
    assert any((_check_get_task_eq(v, expected_task) for v in result.values()))
    assert len(result) < len(dsk)