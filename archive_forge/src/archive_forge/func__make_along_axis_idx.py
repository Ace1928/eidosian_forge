import functools
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, zeros, array, asanyarray
from numpy.core.fromnumeric import reshape, transpose
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from numpy.core import vstack, atleast_3d
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.shape_base import _arrays_for_stack_dispatcher
from numpy.lib.index_tricks import ndindex
from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells
def _make_along_axis_idx(arr_shape, indices, axis):
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError('`indices` and `arr` must have the same number of dimensions')
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))
    return tuple(fancy_index)