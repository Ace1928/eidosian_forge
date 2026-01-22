import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _sputils
def _find_missing_index(ind, n):
    positions = cupy.arange(ind.size)
    diff = ind != positions
    return cupy.where(diff.any(), diff.argmax(), cupy.asarray(ind.size if ind.size < n else -1))