from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
class ArrayBlockwiseDep(BlockwiseDep):
    """
    Blockwise dep for array-likes, which only needs chunking
    information to compute its data.
    """
    chunks: tuple[tuple[int, ...], ...]
    numblocks: tuple[int, ...]
    produces_tasks: bool = False

    def __init__(self, chunks: tuple[tuple[int, ...], ...]):
        self.chunks = chunks
        self.numblocks = tuple((len(chunk) for chunk in chunks))
        self.produces_tasks = False

    def __getitem__(self, idx: tuple[int, ...]):
        raise NotImplementedError('Subclasses must implement __getitem__')