from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def _full_like_variable(other: Variable, fill_value: Any, dtype: DTypeLike | None=None, chunks: T_Chunks=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None) -> Variable:
    """Inner function of full_like, where other must be a variable"""
    from xarray.core.variable import Variable
    if fill_value is dtypes.NA:
        fill_value = dtypes.get_fill_value(dtype if dtype is not None else other.dtype)
    if is_chunked_array(other.data) or chunked_array_type is not None or chunks is not None:
        if chunked_array_type is None:
            chunkmanager = get_chunked_array_type(other.data)
        else:
            chunkmanager = guess_chunkmanager(chunked_array_type)
        if dtype is None:
            dtype = other.dtype
        if from_array_kwargs is None:
            from_array_kwargs = {}
        data = chunkmanager.array_api.full(other.shape, fill_value, dtype=dtype, chunks=chunks if chunks else other.data.chunks, **from_array_kwargs)
    else:
        data = np.full_like(other.data, fill_value, dtype=dtype)
    return Variable(dims=other.dims, data=data, attrs=other.attrs)