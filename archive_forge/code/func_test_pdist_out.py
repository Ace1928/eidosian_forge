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
def test_pdist_out(self, metric):
    eps = 1e-15
    X = eo['random-float32-data'][::5, ::2]
    out_size = int(X.shape[0] * (X.shape[0] - 1) / 2)
    kwargs = dict()
    if metric == 'minkowski':
        kwargs['p'] = 1.23
    out1 = np.empty(out_size, dtype=np.float64)
    Y_right = pdist(X, metric, **kwargs)
    Y_test1 = pdist(X, metric, out=out1, **kwargs)
    assert_allclose(Y_test1, Y_right, rtol=eps)
    assert_(Y_test1 is out1)
    out2 = np.empty(out_size + 3, dtype=np.float64)
    with pytest.raises(ValueError):
        pdist(X, metric, out=out2, **kwargs)
    out3 = np.empty(2 * out_size, dtype=np.float64)[::2]
    with pytest.raises(ValueError):
        pdist(X, metric, out=out3, **kwargs)
    out5 = np.empty(out_size, dtype=np.int64)
    with pytest.raises(ValueError):
        pdist(X, metric, out=out5, **kwargs)