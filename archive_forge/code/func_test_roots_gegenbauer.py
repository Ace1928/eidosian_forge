import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_roots_gegenbauer():

    def rootf(a):
        return lambda n, mu: sc.roots_gegenbauer(n, a, mu)

    def evalf(a):
        return lambda n, x: sc.eval_gegenbauer(n, a, x)

    def weightf(a):
        return lambda x: (1 - x ** 2) ** (a - 0.5)
    vgq = verify_gauss_quad
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1.0, 1.0, 5)
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1.0, 1.0, 25, atol=1e-12)
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1.0, 1.0, 100, atol=1e-11)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1.0, 1.0, 5)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1.0, 1.0, 25, atol=1e-13)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1.0, 1.0, 100, atol=1e-12)
    vgq(rootf(1), evalf(1), weightf(1), -1.0, 1.0, 5)
    vgq(rootf(1), evalf(1), weightf(1), -1.0, 1.0, 25, atol=1e-13)
    vgq(rootf(1), evalf(1), weightf(1), -1.0, 1.0, 100, atol=1e-12)
    vgq(rootf(10), evalf(10), weightf(10), -1.0, 1.0, 5)
    vgq(rootf(10), evalf(10), weightf(10), -1.0, 1.0, 25, atol=1e-13)
    vgq(rootf(10), evalf(10), weightf(10), -1.0, 1.0, 100, atol=1e-12)
    vgq(rootf(50), evalf(50), weightf(50), -1.0, 1.0, 5, atol=1e-13)
    vgq(rootf(50), evalf(50), weightf(50), -1.0, 1.0, 25, atol=1e-12)
    vgq(rootf(50), evalf(50), weightf(50), -1.0, 1.0, 100, atol=1e-11)
    vgq(rootf(170), evalf(170), weightf(170), -1.0, 1.0, 5, atol=1e-13)
    vgq(rootf(170), evalf(170), weightf(170), -1.0, 1.0, 25, atol=1e-12)
    vgq(rootf(170), evalf(170), weightf(170), -1.0, 1.0, 100, atol=1e-11)
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1.0, 1.0, 5, atol=1.25e-13)
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1.0, 1.0, 25, atol=1e-12)
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1.0, 1.0, 100, atol=1e-11)
    vgq(rootf(238), evalf(238), weightf(238), -1.0, 1.0, 5, atol=1e-13)
    vgq(rootf(238), evalf(238), weightf(238), -1.0, 1.0, 25, atol=1e-12)
    vgq(rootf(238), evalf(238), weightf(238), -1.0, 1.0, 100, atol=1e-11)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1.0, 1.0, 5, atol=1e-12)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1.0, 1.0, 25, atol=1e-11)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1.0, 1.0, 100, atol=1e-10)
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1.0, 1.0, 5)
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1.0, 1.0, 25)
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1.0, 1.0, 100, atol=1e-12)
    x, w = sc.roots_gegenbauer(5, 2, False)
    y, v, m = sc.roots_gegenbauer(5, 2, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    muI, muI_err = integrate.quad(weightf(2), -1, 1)
    assert_allclose(m, muI, rtol=muI_err)
    assert_raises(ValueError, sc.roots_gegenbauer, 0, 2)
    assert_raises(ValueError, sc.roots_gegenbauer, 3.3, 2)
    assert_raises(ValueError, sc.roots_gegenbauer, 3, -0.75)