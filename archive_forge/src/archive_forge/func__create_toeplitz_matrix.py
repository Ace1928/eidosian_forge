import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
def _create_toeplitz_matrix(c, r, hankel=False):
    vals = cupy.concatenate((c, r))
    n = vals.strides[0]
    return cupy.lib.stride_tricks.as_strided(vals if hankel else vals[c.size - 1:], shape=(c.size, r.size + 1), strides=(n if hankel else -n, n)).copy()