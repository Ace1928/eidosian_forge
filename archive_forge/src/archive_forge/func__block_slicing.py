import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _block_slicing(arrays, list_ndim, result_ndim):
    shape, slices, arrays = _block_info_recursion(arrays, list_ndim, result_ndim)
    dtype = _nx.result_type(*[arr.dtype for arr in arrays])
    F_order = all((arr.flags['F_CONTIGUOUS'] for arr in arrays))
    C_order = all((arr.flags['C_CONTIGUOUS'] for arr in arrays))
    order = 'F' if F_order and (not C_order) else 'C'
    result = _nx.empty(shape=shape, dtype=dtype, order=order)
    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result