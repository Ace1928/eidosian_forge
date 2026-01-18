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
def to_backend(ddf: _Frame, backend: str | None=None, **kwargs):
    """Move a DataFrame collection to a new backend

    Parameters
    ----------
    ddf : DataFrame, Series, or Index
        The input dataframe collection.
    backend : str, Optional
        The name of the new backend to move to. The default
        is the current "dataframe.backend" configuration.

    Returns
    -------
    dask.DataFrame, dask.Series or dask.Index
        A new dataframe collection with the backend
        specified by ``backend``.
    """
    backend = backend or dataframe_creation_dispatch.backend
    backend_entrypoint = dataframe_creation_dispatch.dispatch(backend)
    return backend_entrypoint.to_backend(ddf, **kwargs)