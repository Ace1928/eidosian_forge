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
def wait_on(*collections, split_every: float | Literal[False] | None=None):
    """Ensure that all chunks of all input collections have been computed before
    computing the dependents of any of the chunks.

    The following example creates a dask array ``u`` that, when used in a computation,
    will only proceed when all chunks of the array ``x`` have been computed, but
    otherwise matches ``x``:

    >>> import dask.array as da
    >>> x = da.ones(10, chunks=5)
    >>> u = wait_on(x)

    The following example will create two arrays ``u`` and ``v`` that, when used in a
    computation, will only proceed when all chunks of the arrays ``x`` and ``y`` have
    been computed but otherwise match ``x`` and ``y``:

    >>> x = da.ones(10, chunks=5)
    >>> y = da.zeros(10, chunks=5)
    >>> u, v = wait_on(x, y)

    Parameters
    ----------
    collections
        Zero or more Dask collections or nested structures of Dask collections
    split_every
        See :func:`checkpoint`

    Returns
    -------
    Same as ``collections``
        Dask collection of the same type as the input, which computes to the same value,
        or a nested structure equivalent to the input where the original collections
        have been replaced.
        The keys of the regenerated nodes of the new collections will be different from
        the original ones, so that they can be used within the same graph.
    """
    blocker = checkpoint(*collections, split_every=split_every)

    def block_one(coll):
        tok = tokenize(coll, blocker)
        dsks = []
        rename = {}
        for prev_name in get_collection_names(coll):
            new_name = 'wait_on-' + tokenize(prev_name, tok)
            rename[prev_name] = new_name
            layer = _build_map_layer(chunks.bind, prev_name, new_name, coll, dependencies=(blocker,))
            dsks.append(HighLevelGraph.from_collections(new_name, layer, dependencies=(coll, blocker)))
        dsk = HighLevelGraph.merge(*dsks)
        rebuild, args = coll.__dask_postpersist__()
        return rebuild(dsk, *args, rename=rename)
    unpacked, repack = unpack_collections(*collections)
    out = repack([block_one(coll) for coll in unpacked])
    return out[0] if len(collections) == 1 else out