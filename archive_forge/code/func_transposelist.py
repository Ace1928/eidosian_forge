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
def transposelist(arrays, axes, extradims=0):
    """Permute axes of nested list

    >>> transposelist([[1,1,1],[1,1,1]], [2,1])
    [[[1, 1], [1, 1], [1, 1]]]

    >>> transposelist([[1,1,1],[1,1,1]], [2,1], extradims=1)
    [[[[1], [1]], [[1], [1]], [[1], [1]]]]
    """
    if len(axes) != ndimlist(arrays):
        raise ValueError('Length of axes should equal depth of nested arrays')
    if extradims < 0:
        raise ValueError('`newdims` should be positive')
    if len(axes) > len(set(axes)):
        raise ValueError('`axes` should be unique')
    ndim = max(axes) + 1
    shape = shapelist(arrays)
    newshape = [shape[axes.index(i)] if i in axes else 1 for i in range(ndim + extradims)]
    result = list(core.flatten(arrays))
    return reshapelist(newshape, result)