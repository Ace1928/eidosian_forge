import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def verify_gauss_quad(root_func, eval_func, weight_func, a, b, N, rtol=1e-15, atol=5e-14):
    x, w, mu = root_func(N, True)
    n = np.arange(N, dtype=np.dtype('long'))
    v = eval_func(n[:, np.newaxis], x)
    vv = np.dot(v * w, v.T)
    vd = 1 / np.sqrt(vv.diagonal())
    vv = vd[:, np.newaxis] * vv * vd
    assert_allclose(vv, np.eye(N), rtol, atol)
    assert_allclose(w.sum(), mu, rtol, atol)

    def f(x):
        return x ** 3 - 3 * x ** 2 + x - 2
    resI = integrate.quad(lambda x: f(x) * weight_func(x), a, b)
    resG = np.vdot(f(x), w)
    rtol = 1e-06 if 1e-06 < resI[1] else resI[1] * 10
    assert_allclose(resI[0], resG, rtol=rtol)