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
def _combine_indexers(old_key, shape: _Shape, new_key) -> VectorizedIndexer:
    """Combine two indexers.

    Parameters
    ----------
    old_key : ExplicitIndexer
        The first indexer for the original array
    shape : tuple of ints
        Shape of the original array to be indexed by old_key
    new_key
        The second indexer for indexing original[old_key]
    """
    if not isinstance(old_key, VectorizedIndexer):
        old_key = _outer_to_vectorized_indexer(old_key, shape)
    if len(old_key.tuple) == 0:
        return new_key
    new_shape = np.broadcast(*old_key.tuple).shape
    if isinstance(new_key, VectorizedIndexer):
        new_key = _arrayize_vectorized_indexer(new_key, new_shape)
    else:
        new_key = _outer_to_vectorized_indexer(new_key, new_shape)
    return VectorizedIndexer(tuple((o[new_key.tuple] for o in np.broadcast_arrays(*old_key.tuple))))