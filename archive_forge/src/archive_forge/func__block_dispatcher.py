import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _block_dispatcher(arrays):
    if type(arrays) is list:
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        yield arrays