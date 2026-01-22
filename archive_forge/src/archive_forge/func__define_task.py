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
def _define_task(self, input_keys, final_task=False):
    if final_task and self.finalize_func:
        outer_func = self.finalize_func
    else:
        outer_func = self.tree_node_func
    return (toolz.pipe, input_keys, self.concat_func, outer_func)