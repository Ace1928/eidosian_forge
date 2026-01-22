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
def _decompose_outer_indexer(indexer: BasicIndexer | OuterIndexer, shape: _Shape, indexing_support: IndexingSupport) -> tuple[ExplicitIndexer, ExplicitIndexer]:
    """
    Decompose outer indexer to the successive two indexers, where the
    first indexer will be used to index backend arrays, while the second one
    is used to index the loaded on-memory np.ndarray.

    Parameters
    ----------
    indexer : OuterIndexer or BasicIndexer
    indexing_support : One of the entries of IndexingSupport

    Returns
    -------
    backend_indexer: OuterIndexer or BasicIndexer
    np_indexers: an ExplicitIndexer (OuterIndexer / BasicIndexer)

    Notes
    -----
    This function is used to realize the vectorized indexing for the backend
    arrays that only support basic or outer indexing.

    As an example, let us consider to index a few elements from a backend array
    with a orthogonal indexer ([0, 3, 1], [2, 3, 2]).
    Even if the backend array only supports basic indexing, it is more
    efficient to load a subslice of the array than loading the entire array,

    >>> array = np.arange(36).reshape(6, 6)
    >>> backend_indexer = BasicIndexer((slice(0, 3), slice(2, 4)))
    >>> # load subslice of the array
    ... array = NumpyIndexingAdapter(array)[backend_indexer]
    >>> np_indexer = OuterIndexer((np.array([0, 2, 1]), np.array([0, 1, 0])))
    >>> # outer indexing for on-memory np.ndarray.
    ... NumpyIndexingAdapter(array).oindex[np_indexer]
    array([[ 2,  3,  2],
           [14, 15, 14],
           [ 8,  9,  8]])
    """
    backend_indexer: list[Any] = []
    np_indexer: list[Any] = []
    assert isinstance(indexer, (OuterIndexer, BasicIndexer))
    if indexing_support == IndexingSupport.VECTORIZED:
        for k, s in zip(indexer.tuple, shape):
            if isinstance(k, slice):
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)
            else:
                backend_indexer.append(k)
                if not is_scalar(k):
                    np_indexer.append(slice(None))
        return (type(indexer)(tuple(backend_indexer)), BasicIndexer(tuple(np_indexer)))
    pos_indexer: list[np.ndarray | int | np.number] = []
    for k, s in zip(indexer.tuple, shape):
        if isinstance(k, np.ndarray):
            pos_indexer.append(np.where(k < 0, k + s, k))
        elif isinstance(k, integer_types) and k < 0:
            pos_indexer.append(k + s)
        else:
            pos_indexer.append(k)
    indexer_elems = pos_indexer
    if indexing_support is IndexingSupport.OUTER_1VECTOR:
        gains = [(np.max(k) - np.min(k) + 1.0) / len(np.unique(k)) if isinstance(k, np.ndarray) else 0 for k in indexer_elems]
        array_index = np.argmax(np.array(gains)) if len(gains) > 0 else None
        for i, (k, s) in enumerate(zip(indexer_elems, shape)):
            if isinstance(k, np.ndarray) and i != array_index:
                backend_indexer.append(slice(np.min(k), np.max(k) + 1))
                np_indexer.append(k - np.min(k))
            elif isinstance(k, np.ndarray):
                pkey, ekey = np.unique(k, return_inverse=True)
                backend_indexer.append(pkey)
                np_indexer.append(ekey)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            else:
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)
        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))
    if indexing_support == IndexingSupport.OUTER:
        for k, s in zip(indexer_elems, shape):
            if isinstance(k, slice):
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            elif isinstance(k, np.ndarray) and (np.diff(k) >= 0).all():
                backend_indexer.append(k)
                np_indexer.append(slice(None))
            else:
                oind, vind = np.unique(k, return_inverse=True)
                backend_indexer.append(oind)
                np_indexer.append(vind.reshape(*k.shape))
        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))
    assert indexing_support == IndexingSupport.BASIC
    for k, s in zip(indexer_elems, shape):
        if isinstance(k, np.ndarray):
            backend_indexer.append(slice(np.min(k), np.max(k) + 1))
            np_indexer.append(k - np.min(k))
        elif isinstance(k, integer_types):
            backend_indexer.append(k)
        else:
            bk_slice, np_slice = _decompose_slice(k, s)
            backend_indexer.append(bk_slice)
            np_indexer.append(np_slice)
    return (BasicIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))