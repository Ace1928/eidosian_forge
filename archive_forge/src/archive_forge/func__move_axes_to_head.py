import collections.abc
import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupy._core._gufuncs import _GUFunc
from cupy.linalg import _solve
from cupy.linalg import _util
def _move_axes_to_head(a, axes):
    for idx, axis in enumerate(axes):
        if idx != axis:
            break
    else:
        return a
    return a.transpose(axes + [i for i in range(a.ndim) if i not in axes])