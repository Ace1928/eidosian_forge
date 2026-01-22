from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def _scalar_binary(op, self, other, inv=False):
    name = f'{funcname(op)}-{tokenize(self, other)}'
    dependencies = [self]
    dsk = {}
    return_type = get_parallel_type(other)
    if isinstance(other, Scalar):
        dependencies.append(other)
        other_key = (other._name, 0)
    elif is_dask_collection(other):
        return NotImplemented
    else:
        other_key = other
    dsk[name, 0] = (op, other_key, (self._name, 0)) if inv else (op, (self._name, 0), other_key)
    other_meta = make_meta(other, parent_meta=self._parent_meta)
    other_meta_nonempty = meta_nonempty(other_meta)
    if inv:
        meta = op(other_meta_nonempty, self._meta_nonempty)
    else:
        meta = op(self._meta_nonempty, other_meta_nonempty)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    if return_type is not Scalar:
        return return_type(graph, name, meta, [other.index.min(), other.index.max()])
    else:
        return Scalar(graph, name, meta)