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
def test_nonfusible_fancy_indexing():
    nil = slice(None)
    cases = [((nil, [1, 2, 3], nil), (0, nil, nil)), ((0, nil, nil), (nil, [1, 2, 3], nil)), ((nil, [1, 2], nil, nil), (nil, nil, nil, 0))]
    for a, b in cases:
        with pytest.raises(NotImplementedError):
            fuse_slice(a, b)