from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
class CopyOnWriteArray(ExplicitlyIndexedNDArrayMixin):
    __slots__ = ('array', '_copied')

    def __init__(self, array: duckarray[Any, Any]):
        self.array = as_indexable(array)
        self._copied = False

    def _ensure_copied(self):
        if not self._copied:
            self.array = as_indexable(np.array(self.array))
            self._copied = True

    def get_duck_array(self):
        return self.array.get_duck_array()

    def _oindex_get(self, indexer: OuterIndexer):
        return type(self)(_wrap_numpy_scalars(self.array.oindex[indexer]))

    def _vindex_get(self, indexer: VectorizedIndexer):
        return type(self)(_wrap_numpy_scalars(self.array.vindex[indexer]))

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return type(self)(_wrap_numpy_scalars(self.array[indexer]))

    def transpose(self, order):
        return self.array.transpose(order)

    def _vindex_set(self, indexer: VectorizedIndexer, value: Any) -> None:
        self._ensure_copied()
        self.array.vindex[indexer] = value

    def _oindex_set(self, indexer: OuterIndexer, value: Any) -> None:
        self._ensure_copied()
        self.array.oindex[indexer] = value

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        self._ensure_copied()
        self.array[indexer] = value

    def __deepcopy__(self, memo):
        return type(self)(self.array)