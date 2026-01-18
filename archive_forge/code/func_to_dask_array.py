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
def to_dask_array(self, lengths=None, meta=None):
    """Convert a dask DataFrame to a dask array.

        Parameters
        ----------
        lengths : bool or Sequence of ints, optional
            How to determine the chunks sizes for the output array.
            By default, the output array will have unknown chunk lengths
            along the first axis, which can cause some later operations
            to fail.

            * True : immediately compute the length of each partition
            * Sequence : a sequence of integers to use for the chunk sizes
              on the first axis. These values are *not* validated for
              correctness, beyond ensuring that the number of items
              matches the number of partitions.
        meta : object, optional
            An optional `meta` parameter can be passed for dask to override the
            default metadata on the underlying dask array.

        Returns
        -------
        """
    if lengths is True:
        lengths = tuple(self.map_partitions(len, enforce_metadata=False).compute())
    arr = self.values
    chunks = self._validate_chunks(arr, lengths)
    arr._chunks = chunks
    if meta is not None:
        arr._meta = meta
    return arr