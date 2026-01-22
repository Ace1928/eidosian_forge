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
def _make_api(attr):

    def wrapper(*args, **kwargs):
        key = array_creation_dispatch.backend
        with _cached_states_lock:
            try:
                state = _cached_states[key]
            except KeyError:
                _cached_states[key] = state = RandomState()
        return getattr(state, attr)(*args, **kwargs)
    wrapper.__name__ = getattr(RandomState, attr).__name__
    wrapper.__doc__ = getattr(RandomState, attr).__doc__
    return wrapper