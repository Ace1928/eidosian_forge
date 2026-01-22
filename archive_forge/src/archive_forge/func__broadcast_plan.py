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
@property
def _broadcast_plan(self):
    if self.lhs_npartitions < self.rhs_npartitions:
        return (self.lhs_name, self.lhs_npartitions, self.rhs_name, self.right_on)
    else:
        return (self.rhs_name, self.rhs_npartitions, self.lhs_name, self.left_on)