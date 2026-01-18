import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_roots_jacobi():

    def rf(a, b):
        return lambda n, mu: sc.roots_jacobi(n, a, b, mu)

    def ef(a, b):
        return lambda n, x: sc.eval_jacobi(n, a, b, x)

    def wf(a, b):
        return lambda x: (1 - x) ** a * (1 + x) ** b
    vgq = verify_gauss_quad
    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1.0, 1.0, 5)
    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1.0, 1.0, 25, atol=1e-12)
    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1.0, 1.0, 100, atol=1e-11)
    vgq(rf(0.5, -0.5), ef(0.5, -0.5), wf(0.5, -0.5), -1.0, 1.0, 5)
    vgq(rf(0.5, -0.5), ef(0.5, -0.5), wf(0.5, -0.5), -1.0, 1.0, 25, atol=1.5e-13)
    vgq(rf(0.5, -0.5), ef(0.5, -0.5), wf(0.5, -0.5), -1.0, 1.0, 100, atol=2e-12)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), -1.0, 1.0, 5, atol=2e-13)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), -1.0, 1.0, 25, atol=2e-13)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), -1.0, 1.0, 100, atol=1e-12)
    vgq(rf(0.9, 2), ef(0.9, 2), wf(0.9, 2), -1.0, 1.0, 5)
    vgq(rf(0.9, 2), ef(0.9, 2), wf(0.9, 2), -1.0, 1.0, 25, atol=1e-13)
    vgq(rf(0.9, 2), ef(0.9, 2), wf(0.9, 2), -1.0, 1.0, 100, atol=3e-13)
    vgq(rf(18.24, 27.3), ef(18.24, 27.3), wf(18.24, 27.3), -1.0, 1.0, 5)
    vgq(rf(18.24, 27.3), ef(18.24, 27.3), wf(18.24, 27.3), -1.0, 1.0, 25, atol=1.1e-14)
    vgq(rf(18.24, 27.3), ef(18.24, 27.3), wf(18.24, 27.3), -1.0, 1.0, 100, atol=1e-13)
    vgq(rf(47.1, -0.2), ef(47.1, -0.2), wf(47.1, -0.2), -1.0, 1.0, 5, atol=1e-13)
    vgq(rf(47.1, -0.2), ef(47.1, -0.2), wf(47.1, -0.2), -1.0, 1.0, 25, atol=2e-13)
    vgq(rf(47.1, -0.2), ef(47.1, -0.2), wf(47.1, -0.2), -1.0, 1.0, 100, atol=1e-11)
    vgq(rf(1.0, 658.0), ef(1.0, 658.0), wf(1.0, 658.0), -1.0, 1.0, 5, atol=2e-13)
    vgq(rf(1.0, 658.0), ef(1.0, 658.0), wf(1.0, 658.0), -1.0, 1.0, 25, atol=1e-12)
    vgq(rf(1.0, 658.0), ef(1.0, 658.0), wf(1.0, 658.0), -1.0, 1.0, 100, atol=1e-11)
    vgq(rf(1.0, 658.0), ef(1.0, 658.0), wf(1.0, 658.0), -1.0, 1.0, 250, atol=1e-11)
    vgq(rf(511.0, 511.0), ef(511.0, 511.0), wf(511.0, 511.0), -1.0, 1.0, 5, atol=1e-12)
    vgq(rf(511.0, 511.0), ef(511.0, 511.0), wf(511.0, 511.0), -1.0, 1.0, 25, atol=1e-11)
    vgq(rf(511.0, 511.0), ef(511.0, 511.0), wf(511.0, 511.0), -1.0, 1.0, 100, atol=1e-10)
    vgq(rf(511.0, 512.0), ef(511.0, 512.0), wf(511.0, 512.0), -1.0, 1.0, 5, atol=1e-12)
    vgq(rf(511.0, 512.0), ef(511.0, 512.0), wf(511.0, 512.0), -1.0, 1.0, 25, atol=1e-11)
    vgq(rf(511.0, 512.0), ef(511.0, 512.0), wf(511.0, 512.0), -1.0, 1.0, 100, atol=1e-10)
    vgq(rf(1000.0, 500.0), ef(1000.0, 500.0), wf(1000.0, 500.0), -1.0, 1.0, 5, atol=1e-12)
    vgq(rf(1000.0, 500.0), ef(1000.0, 500.0), wf(1000.0, 500.0), -1.0, 1.0, 25, atol=1e-11)
    vgq(rf(1000.0, 500.0), ef(1000.0, 500.0), wf(1000.0, 500.0), -1.0, 1.0, 100, atol=1e-10)
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1.0, 1.0, 5)
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1.0, 1.0, 25, atol=1e-13)
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1.0, 1.0, 100, atol=1e-13)
    xj, wj = sc.roots_jacobi(6, 0.0, 0.0)
    xl, wl = sc.roots_legendre(6)
    assert_allclose(xj, xl, 1e-14, 1e-14)
    assert_allclose(wj, wl, 1e-14, 1e-14)
    xj, wj = sc.roots_jacobi(6, 4.0, 4.0)
    xc, wc = sc.roots_gegenbauer(6, 4.5)
    assert_allclose(xj, xc, 1e-14, 1e-14)
    assert_allclose(wj, wc, 1e-14, 1e-14)
    x, w = sc.roots_jacobi(5, 2, 3, False)
    y, v, m = sc.roots_jacobi(5, 2, 3, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    muI, muI_err = integrate.quad(wf(2, 3), -1, 1)
    assert_allclose(m, muI, rtol=muI_err)
    assert_raises(ValueError, sc.roots_jacobi, 0, 1, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3.3, 1, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3, -2, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3, 1, -2)
    assert_raises(ValueError, sc.roots_jacobi, 3, -2, -2)