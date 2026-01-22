from __future__ import annotations
from collections.abc import Iterable
from functools import partial
from math import ceil
from operator import getitem
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
import pandas as pd
import dask.array as da
from dask.base import is_dask_collection, tokenize
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.dataframe._compat import is_any_real_numeric_dtype
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import (
from dask.dataframe.dispatch import meta_lib_from_array
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import M, funcname, is_arraylike
def _partition_from_array(data, index=None, initializer=None, **kwargs):
    """Create a Dask partition for either a DataFrame or Series.

    Designed to be used with :func:`dask.blockwise.blockwise`. ``data`` is the array
    from which the partition will be created. ``index`` can be:

    1. ``None``, in which case each partition has an independent RangeIndex
    2. a `tuple` with two elements, the start and stop values for a RangeIndex for
       this partition, which gives a continuously varying RangeIndex over the
       whole Dask DataFrame
    3. an instance of a ``pandas.Index`` or a subclass thereof

    The ``kwargs`` _must_ contain an ``initializer`` key which is set by calling
    ``type(meta)``.
    """
    if isinstance(index, tuple):
        index = pd.RangeIndex(*index)
    return initializer(data, index=index, **kwargs)