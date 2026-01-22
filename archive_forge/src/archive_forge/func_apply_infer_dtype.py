from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def apply_infer_dtype(func, args, kwargs, funcname, suggest_dtype='dtype', nout=None):
    """
    Tries to infer output dtype of ``func`` for a small set of input arguments.

    Parameters
    ----------
    func: Callable
        Function for which output dtype is to be determined

    args: List of array like
        Arguments to the function, which would usually be used. Only attributes
        ``ndim`` and ``dtype`` are used.

    kwargs: dict
        Additional ``kwargs`` to the ``func``

    funcname: String
        Name of calling function to improve potential error messages

    suggest_dtype: None/False or String
        If not ``None`` adds suggestion to potential error message to specify a dtype
        via the specified kwarg. Defaults to ``'dtype'``.

    nout: None or Int
        ``None`` if function returns single output, integer if many.
        Defaults to ``None``.

    Returns
    -------
    : dtype or List of dtype
        One or many dtypes (depending on ``nout``)
    """
    from dask.array.utils import meta_from_array
    args = [np.ones_like(meta_from_array(x), shape=(1,) * x.ndim, dtype=x.dtype) if is_arraylike(x) else x for x in args]
    try:
        with np.errstate(all='ignore'):
            o = func(*args, **kwargs)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = ''.join(traceback.format_tb(exc_traceback))
        suggest = 'Please specify the dtype explicitly using the `{dtype}` kwarg.\n\n'.format(dtype=suggest_dtype) if suggest_dtype else ''
        msg = f'`dtype` inference failed in `{funcname}`.\n\n{suggest}Original error is below:\n------------------------\n{e!r}\n\nTraceback:\n---------\n{tb}'
    else:
        msg = None
    if msg is not None:
        raise ValueError(msg)
    return getattr(o, 'dtype', type(o)) if nout is None else tuple((e.dtype for e in o))