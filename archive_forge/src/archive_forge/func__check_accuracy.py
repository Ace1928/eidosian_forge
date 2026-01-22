import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def _check_accuracy(self, func, x=None, tol=1e-06, alternate=False, rescale=False, **kw):
    np.random.seed(1234)
    if x is None:
        x = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8), (0.5, 0.2)], dtype=float)
    if not alternate:
        ip = interpnd.CloughTocher2DInterpolator(x, func(x[:, 0], x[:, 1]), tol=1e-06, rescale=rescale)
    else:
        ip = interpnd.CloughTocher2DInterpolator((x[:, 0], x[:, 1]), func(x[:, 0], x[:, 1]), tol=1e-06, rescale=rescale)
    p = np.random.rand(50, 2)
    if not alternate:
        a = ip(p)
    else:
        a = ip(p[:, 0], p[:, 1])
    b = func(p[:, 0], p[:, 1])
    try:
        assert_allclose(a, b, **kw)
    except AssertionError:
        print('_check_accuracy: abs(a-b):', abs(a - b))
        print('ip.grad:', ip.grad)
        raise