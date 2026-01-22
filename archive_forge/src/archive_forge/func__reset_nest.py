import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def _reset_nest(self, data, nest=None):
    n = data[10]
    if nest is None:
        k, m = (data[5], len(data[0]))
        nest = m + k + 1
    elif not n <= nest:
        raise ValueError('`nest` can only be increased')
    t, c, fpint, nrdata = (np.resize(data[j], nest) for j in [8, 9, 11, 12])
    args = data[:8] + (t, c, n, fpint, nrdata, data[13])
    data = dfitpack.fpcurf1(*args)
    return data