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
def test_sqeuclidean_dtypes():
    x = [1, 2, 3]
    y = [4, 5, 6]
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_(np.issubdtype(d.dtype, np.floating))
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        umax = np.iinfo(dtype).max
        d1 = wsqeuclidean([0], np.asarray([umax], dtype=dtype))
        d2 = wsqeuclidean(np.asarray([umax], dtype=dtype), [0])
        assert_equal(d1, d2)
        assert_equal(d1, np.float64(umax) ** 2)
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for dtype in ['float16', 'float128']:
        if hasattr(np, dtype):
            dtypes.append(getattr(np, dtype))
    for dtype in dtypes:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_equal(d.dtype, dtype)