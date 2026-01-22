from __future__ import annotations
import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar
import numpy as np
from xarray.core.utils import emit_user_level_warning
from xarray.namedarray.pycompat import is_chunked_array
class ChunkedArrayMixinProtocol(Protocol):

    def rechunk(self, chunks: Any, **kwargs: Any) -> Any:
        ...

    @property
    def dtype(self) -> np.dtype[Any]:
        ...

    @property
    def chunks(self) -> _NormalizedChunks:
        ...

    def compute(self, *data: Any, **kwargs: Any) -> tuple[np.ndarray[Any, _DType_co], ...]:
        ...