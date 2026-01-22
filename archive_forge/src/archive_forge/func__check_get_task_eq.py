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
def _check_get_task_eq(a, b) -> bool:
    """
    Check that two tasks (possibly containing nested tasks) are equal, where
    equality is lax by allowing the callable in a SubgraphCallable to be the same
    as a non-wrapped task.
    """
    if len(a) < 1 or len(a) != len(b):
        return False
    a_callable = list(a[0].dsk.values())[0][0] if isinstance(a[0], SubgraphCallable) else a[0]
    b_callable = list(b[0].dsk.values())[0][0] if isinstance(b[0], SubgraphCallable) else b[0]
    if a_callable != b_callable:
        return False
    for ae, be in zip(a[1:], b[1:]):
        if dask.core.istask(ae):
            if not _check_get_task_eq(ae, be):
                return False
        elif ae != be:
            return False
    return True