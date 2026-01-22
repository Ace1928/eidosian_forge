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
class _PackedArgCallable(DataFrameIOFunction):
    """Packed-argument wrapper for DataFrameIOFunction

    This is a private helper class for ``from_map``. This class
    ensures that packed positional arguments will be expanded
    before the underlying function (``func``) is called. This class
    also handles optional metadata enforcement and column projection
    (when ``func`` satisfies the ``DataFrameIOFunction`` protocol).
    """

    def __init__(self, func, args=None, kwargs=None, meta=None, packed=False, enforce_metadata=False):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.meta = meta
        self.enforce_metadata = enforce_metadata
        self.packed = packed
        self.is_dataframe_io_func = isinstance(self.func, DataFrameIOFunction)

    @property
    def columns(self):
        if self.is_dataframe_io_func:
            return self.func.columns
        return None

    def project_columns(self, columns):
        if self.is_dataframe_io_func:
            return _PackedArgCallable(self.func.project_columns(columns), args=self.args, kwargs=self.kwargs, meta=self.meta[columns], packed=self.packed, enforce_metadata=self.enforce_metadata)
        return self

    def __call__(self, packed_arg):
        if not self.packed:
            packed_arg = [packed_arg]
        if self.enforce_metadata:
            return apply_and_enforce(*packed_arg, *(self.args or []), _func=self.func, _meta=self.meta, **self.kwargs or {})
        return self.func(*packed_arg, *(self.args or []), **self.kwargs or {})