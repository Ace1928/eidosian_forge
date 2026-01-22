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
def _keys_to_output_partitions(self, keys):
    """Simple utility to convert keys to output partition indices."""
    splits = set()
    for key in keys:
        try:
            _name, _split = key
        except ValueError:
            continue
        if _name != self.name:
            continue
        splits.add(_split)
    return splits