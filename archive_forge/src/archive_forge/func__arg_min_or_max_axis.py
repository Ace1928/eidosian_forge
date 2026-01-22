import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _arg_min_or_max_axis(self, axis, op):
    if self.shape[axis] == 0:
        raise ValueError("Can't apply the operation along a zero-sized dimension.")
    mat = self.tocsc() if axis == 0 else self.tocsr()
    mat.sum_duplicates()
    value = mat._arg_minor_reduce(op, axis)
    if axis == 0:
        return value[None, :]
    else:
        return value[:, None]