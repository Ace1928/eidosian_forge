from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
class SubgraphCallable:
    """Create a callable object from a dask graph.

    Parameters
    ----------
    dsk : dict
        A dask graph
    outkey : Dask key
        The output key from the graph
    inkeys : list
        A list of keys to be used as arguments to the callable.
    name : str, optional
        The name to use for the function.
    """
    dsk: Graph
    outkey: Key
    inkeys: tuple[Key, ...]
    name: str
    __slots__ = tuple(__annotations__)

    def __init__(self, dsk: Graph, outkey: Key, inkeys: Iterable[Key], name: str | None=None):
        self.dsk = dsk
        self.outkey = outkey
        self.inkeys = tuple(inkeys)
        if name is None:
            name = 'subgraph_callable-' + tokenize(dsk, outkey, self.inkeys)
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.name == other.name and (self.outkey == other.outkey) and (set(self.inkeys) == set(other.inkeys))

    def __call__(self, *args: Any) -> Any:
        if not len(args) == len(self.inkeys):
            raise ValueError('Expected %d args, got %d' % (len(self.inkeys), len(args)))
        return core.get(self.dsk, self.outkey, dict(zip(self.inkeys, args)))

    def __reduce__(self) -> tuple:
        return (SubgraphCallable, (self.dsk, self.outkey, self.inkeys, self.name))

    def __hash__(self) -> int:
        return hash((self.outkey, frozenset(self.inkeys), self.name))

    def __dask_tokenize__(self) -> object:
        return ('SubgraphCallable', normalize_token(self.dsk), self.outkey, self.inkeys, self.name)