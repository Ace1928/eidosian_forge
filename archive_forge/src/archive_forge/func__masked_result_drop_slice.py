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
def _masked_result_drop_slice(key, data: duckarray[Any, Any] | None=None):
    key = (k for k in key if not isinstance(k, slice))
    chunks_hint = getattr(data, 'chunks', None)
    new_keys = []
    for k in key:
        if isinstance(k, np.ndarray):
            if is_chunked_array(data):
                chunkmanager = get_chunked_array_type(data)
                new_keys.append(_chunked_array_with_chunks_hint(k, chunks_hint, chunkmanager))
            elif isinstance(data, array_type('sparse')):
                import sparse
                new_keys.append(sparse.COO.from_numpy(k))
            else:
                new_keys.append(k)
        else:
            new_keys.append(k)
    mask = _logical_any((k == -1 for k in new_keys))
    return mask