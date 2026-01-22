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
def _apply_random(RandomState, funcname, state_data, size, args, kwargs):
    """Apply RandomState method with seed"""
    if RandomState is None:
        RandomState = array_creation_dispatch.RandomState
    state = RandomState(state_data)
    func = getattr(state, funcname)
    return func(*args, size=size, **kwargs)