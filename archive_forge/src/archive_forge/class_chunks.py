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
class chunks:
    """Callables to be inserted in the Dask graph"""

    @staticmethod
    def bind(node: T, *args, **kwargs) -> T:
        """Dummy graph node of :func:`bind` and :func:`wait_on`.
        Wait for both node and all variadic args to complete; then return node.
        """
        return node

    @staticmethod
    def checkpoint(*args, **kwargs) -> None:
        """Dummy graph node of :func:`checkpoint`.
        Wait for all variadic args to complete; then return None.
        """
        pass