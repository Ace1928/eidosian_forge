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
def to_bag(df, index=False, format='tuple'):
    """Create Dask Bag from a Dask DataFrame

    Parameters
    ----------
    index : bool, optional
        If True, the elements are tuples of ``(index, value)``, otherwise
        they're just the ``value``.  Default is False.
    format : {"tuple", "dict", "frame"}, optional
        Whether to return a bag of tuples, dictionaries, or
        dataframe-like objects. Default is "tuple". If "frame",
        the original partitions of ``df`` will not be transformed
        in any way.


    Examples
    --------
    >>> bag = df.to_bag()  # doctest: +SKIP
    """
    from dask.bag.core import Bag
    if not isinstance(df, (DataFrame, Series)):
        raise TypeError('df must be either DataFrame or Series')
    name = 'to_bag-' + tokenize(df, index, format)
    if format == 'frame':
        dsk = df.dask
        name = df._name
    else:
        dsk = {(name, i): (_df_to_bag, block, index, format) for i, block in enumerate(df.__dask_keys__())}
        dsk.update(df.__dask_optimize__(df.__dask_graph__(), df.__dask_keys__()))
    return Bag(dsk, name, df.npartitions)