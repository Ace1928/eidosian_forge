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
def test_striding(self, metric):
    eps = 1e-15
    X = eo['random-float32-data'][::5, ::2]
    X_copy = X.copy()
    assert_(not X.flags.c_contiguous)
    assert_(X_copy.flags.c_contiguous)
    kwargs = dict()
    if metric == 'minkowski':
        kwargs['p'] = 1.23
    Y1 = pdist(X, metric, **kwargs)
    Y2 = pdist(X_copy, metric, **kwargs)
    assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)