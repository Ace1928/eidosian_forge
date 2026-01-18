from __future__ import annotations
import functools
from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array
from .._core.internal import _normalize_axis_indices as normalize_axis_tuple
from typing import TYPE_CHECKING
from typing import NamedTuple
import cupy as np
def vector_norm(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False, ord: Optional[Union[int, float]]=2) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')
    a = x._array
    if axis is None:
        a = a.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple((i for i in range(a.ndim) if i not in normalized_axis))
        newshape = axis + rest
        a = np.transpose(a, newshape).reshape((np.prod([a.shape[i] for i in axis], dtype=int), *[a.shape[i] for i in rest]))
        _axis = 0
    else:
        _axis = axis
    res = Array._new(np.linalg.norm(a, axis=_axis, ord=ord))
    if keepdims:
        shape = list(x.shape)
        _axis = normalize_axis_tuple(range(x.ndim) if axis is None else axis, x.ndim)
        for i in _axis:
            shape[i] = 1
        res = reshape(res, tuple(shape))
    return res