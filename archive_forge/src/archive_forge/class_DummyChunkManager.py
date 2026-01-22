from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
class DummyChunkManager(ChunkManagerEntrypoint):
    """Mock-up of ChunkManager class for DummyChunkedArray"""

    def __init__(self):
        self.array_cls = DummyChunkedArray

    def is_chunked_array(self, data: Any) -> bool:
        return isinstance(data, DummyChunkedArray)

    def chunks(self, data: DummyChunkedArray) -> T_NormalizedChunks:
        return data.chunks

    def normalize_chunks(self, chunks: T_Chunks | T_NormalizedChunks, shape: tuple[int, ...] | None=None, limit: int | None=None, dtype: np.dtype | None=None, previous_chunks: T_NormalizedChunks | None=None) -> T_NormalizedChunks:
        from dask.array.core import normalize_chunks
        return normalize_chunks(chunks, shape, limit, dtype, previous_chunks)

    def from_array(self, data: T_DuckArray | np.typing.ArrayLike, chunks: _Chunks, **kwargs) -> DummyChunkedArray:
        from dask import array as da
        return da.from_array(data, chunks, **kwargs)

    def rechunk(self, data: DummyChunkedArray, chunks, **kwargs) -> DummyChunkedArray:
        return data.rechunk(chunks, **kwargs)

    def compute(self, *data: DummyChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
        from dask.array import compute
        return compute(*data, **kwargs)

    def apply_gufunc(self, func, signature, *args, axes=None, axis=None, keepdims=False, output_dtypes=None, output_sizes=None, vectorize=None, allow_rechunk=False, meta=None, **kwargs):
        from dask.array.gufunc import apply_gufunc
        return apply_gufunc(func, signature, *args, axes=axes, axis=axis, keepdims=keepdims, output_dtypes=output_dtypes, output_sizes=output_sizes, vectorize=vectorize, allow_rechunk=allow_rechunk, meta=meta, **kwargs)