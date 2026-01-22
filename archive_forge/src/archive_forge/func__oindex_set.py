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
def _oindex_set(self, indexer: OuterIndexer, value: Any) -> None:
    num_non_slices = sum((0 if isinstance(k, slice) else 1 for k in indexer.tuple))
    if num_non_slices > 1:
        raise NotImplementedError("xarray can't set arrays with multiple array indices to dask yet.")
    self.array[indexer.tuple] = value