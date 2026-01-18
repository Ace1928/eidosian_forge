import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_roots_sh_jacobi():

    def rf(a, b):
        return lambda n, mu: sc.roots_sh_jacobi(n, a, b, mu)

    def ef(a, b):
        return lambda n, x: sc.eval_sh_jacobi(n, a, b, x)

    def wf(a, b):
        return lambda x: (1.0 - x) ** (a - b) * x ** (b - 1.0)
    vgq = verify_gauss_quad
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0.0, 1.0, 5)
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0.0, 1.0, 25, atol=1e-12)
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0.0, 1.0, 100, atol=1e-11)
    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0.0, 1.0, 5)
    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0.0, 1.0, 25, atol=1e-13)
    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0.0, 1.0, 100, atol=1e-12)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0.0, 1.0, 5)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0.0, 1.0, 25, atol=1.5e-13)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0.0, 1.0, 100, atol=2e-12)
    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0.0, 1.0, 5)
    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0.0, 1.0, 25, atol=1e-13)
    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0.0, 1.0, 100, atol=1e-12)
    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0.0, 1.0, 5)
    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0.0, 1.0, 25)
    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0.0, 1.0, 100, atol=1e-13)
    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0.0, 1.0, 5, atol=1e-12)
    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0.0, 1.0, 25, atol=1e-11)
    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0.0, 1.0, 100, atol=1e-10)
    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0.0, 1.0, 5, atol=3.5e-14)
    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0.0, 1.0, 25, atol=2e-13)
    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0.0, 1.0, 100, atol=1e-12)
    x, w = sc.roots_sh_jacobi(5, 3, 2, False)
    y, v, m = sc.roots_sh_jacobi(5, 3, 2, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    muI, muI_err = integrate.quad(wf(3, 2), 0, 1)
    assert_allclose(m, muI, rtol=muI_err)
    assert_raises(ValueError, sc.roots_sh_jacobi, 0, 1, 1)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3.3, 1, 1)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, 1, 2)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, 2, -1)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, -2, -1)