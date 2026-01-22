import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def _get_py_shape(x):
    """Infer the shape of a possibly nested list/tuple object."""
    if hasattr(x, 'shape'):
        return tuple(x.shape)
    if isinstance(x, (tuple, list)):
        return (len(x),) + _get_py_shape(x[0])
    return ()