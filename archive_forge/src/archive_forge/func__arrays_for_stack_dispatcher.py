import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _arrays_for_stack_dispatcher(arrays):
    if not hasattr(arrays, '__getitem__'):
        raise TypeError('arrays to stack must be passed as a "sequence" type such as list or tuple.')
    return tuple(arrays)