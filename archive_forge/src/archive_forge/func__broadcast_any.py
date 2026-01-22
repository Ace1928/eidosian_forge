from __future__ import annotations
import contextlib
import importlib
import numbers
from itertools import chain, product
from numbers import Integral
from operator import getitem
from threading import Lock
import numpy as np
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.creation import arange
from dask.array.utils import asarray_safe
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, random_state_data, typename
def _broadcast_any(ar, shape, chunks):
    if isinstance(ar, Array):
        return broadcast_to(ar, shape).rechunk(chunks)
    elif isinstance(ar, np.ndarray):
        return np.ascontiguousarray(np.broadcast_to(ar, shape))
    else:
        raise TypeError('Unknown object type for broadcast')