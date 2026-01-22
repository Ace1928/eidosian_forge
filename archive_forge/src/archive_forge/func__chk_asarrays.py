import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _chk_asarrays(arrays, axis=None):
    arrays = [np.asanyarray(a) for a in arrays]
    if axis is None:
        arrays = [np.ravel(a) if a.ndim != 1 else a for a in arrays]
        axis = 0
    arrays = tuple((np.atleast_1d(a) for a in arrays))
    if axis < 0:
        if not all((a.ndim == arrays[0].ndim for a in arrays)):
            raise ValueError('array ndim must be the same for neg axis')
        axis = range(arrays[0].ndim)[axis]
    return arrays + (axis,)