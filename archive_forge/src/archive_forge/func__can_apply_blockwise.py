from __future__ import annotations
import uuid
from collections.abc import Callable, Hashable
from typing import Literal, TypeVar
from dask.base import (
from dask.blockwise import blockwise
from dask.core import flatten
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer
from dask.typing import Graph, Key
def _can_apply_blockwise(collection) -> bool:
    """Return True if _map_blocks can be sped up via blockwise operations; False
    otherwise.

    FIXME this returns False for collections that wrap around around da.Array, such as
          pint.Quantity, xarray DataArray, Dataset, and Variable.
    """
    try:
        from dask.bag import Bag
        if isinstance(collection, Bag):
            return True
    except ImportError:
        pass
    try:
        from dask.array import Array
        if isinstance(collection, Array):
            return True
    except ImportError:
        pass
    try:
        from dask.dataframe import DataFrame, Series
        return isinstance(collection, (DataFrame, Series))
    except ImportError:
        return False