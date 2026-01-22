from __future__ import annotations
import itertools
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any
import tlz as toolz
import dask
from dask.base import clone_key, get_name_from_key, tokenize
from dask.core import flatten, ishashable, keys_in_tasks, reverse_dict
from dask.highlevelgraph import HighLevelGraph, Layer
from dask.optimization import SubgraphCallable, fuse
from dask.typing import Graph, Key
from dask.utils import (
class BlockIndex(BlockwiseDep):
    """Index BlockwiseDep argument

    The purpose of this class is to provide each
    block of a ``Blockwise``-based operation with
    the current block index.
    """
    produces_tasks: bool = False

    def __init__(self, numblocks: tuple[int, ...]):
        self.numblocks = numblocks

    def __getitem__(self, idx: tuple[int, ...]) -> tuple[int, ...]:
        return idx