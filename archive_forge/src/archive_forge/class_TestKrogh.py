import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
class TestKrogh:

    def setup_method(self):
        self.true_poly = np.polynomial.Polynomial([-4, 5, 1, 3, -2])
        self.test_xs = np.linspace(-1, 1, 100)
        self.xs = np.linspace(-1, 1, 5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = KroghInterpolator(self.xs, self.ys)
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_scalar(self):
        P = KroghInterpolator(self.xs, self.ys)
        assert_almost_equal(self.true_poly(7), P(7))
        assert_almost_equal(self.true_poly(np.array(7)), P(np.array(7)))

    def test_derivatives(self):
        P = KroghInterpolator(self.xs, self.ys)
        D = P.derivatives(self.test_xs)
        for i in range(D.shape[0]):
            assert_almost_equal(self.true_poly.deriv(i)(self.test_xs), D[i])

    def test_low_derivatives(self):
        P = KroghInterpolator(self.xs, self.ys)
        D = P.derivatives(self.test_xs, len(self.xs) + 2)
        for i in range(D.shape[0]):
            assert_almost_equal(self.true_poly.deriv(i)(self.test_xs), D[i])

    def test_derivative(self):
        P = KroghInterpolator(self.xs, self.ys)
        m = 10
        r = P.derivatives(self.test_xs, m)
        for i in range(m):
            assert_almost_equal(P.derivative(self.test_xs, i), r[i])

    def test_high_derivative(self):
        P = KroghInterpolator(self.xs, self.ys)
        for i in range(len(self.xs), 2 * len(self.xs)):
            assert_almost_equal(P.derivative(self.test_xs, i), np.zeros(len(self.test_xs)))

    def test_ndim_derivatives(self):
        poly1 = self.true_poly
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
        P = KroghInterpolator(self.xs, ys, axis=0)
        D = P.derivatives(self.test_xs)
        for i in range(D.shape[0]):
            assert_allclose(D[i], np.stack((poly1.deriv(i)(self.test_xs), poly2.deriv(i)(self.test_xs), poly3.deriv(i)(self.test_xs)), axis=-1))

    def test_ndim_derivative(self):
        poly1 = self.true_poly
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
        P = KroghInterpolator(self.xs, ys, axis=0)
        for i in range(P.n):
            assert_allclose(P.derivative(self.test_xs, i), np.stack((poly1.deriv(i)(self.test_xs), poly2.deriv(i)(self.test_xs), poly3.deriv(i)(self.test_xs)), axis=-1))

    def test_hermite(self):
        P = KroghInterpolator(self.xs, self.ys)
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_vector(self):
        xs = [0, 1, 2]
        ys = np.array([[0, 1], [1, 0], [2, 1]])
        P = KroghInterpolator(xs, ys)
        Pi = [KroghInterpolator(xs, ys[:, i]) for i in range(ys.shape[1])]
        test_xs = np.linspace(-1, 3, 100)
        assert_almost_equal(P(test_xs), np.asarray([p(test_xs) for p in Pi]).T)
        assert_almost_equal(P.derivatives(test_xs), np.transpose(np.asarray([p.derivatives(test_xs) for p in Pi]), (1, 2, 0)))

    def test_empty(self):
        P = KroghInterpolator(self.xs, self.ys)
        assert_array_equal(P([]), [])

    def test_shapes_scalarvalue(self):
        P = KroghInterpolator(self.xs, self.ys)
        assert_array_equal(np.shape(P(0)), ())
        assert_array_equal(np.shape(P(np.array(0))), ())
        assert_array_equal(np.shape(P([0])), (1,))
        assert_array_equal(np.shape(P([0, 1])), (2,))

    def test_shapes_scalarvalue_derivative(self):
        P = KroghInterpolator(self.xs, self.ys)
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n,))
        assert_array_equal(np.shape(P.derivatives(np.array(0))), (n,))
        assert_array_equal(np.shape(P.derivatives([0])), (n, 1))
        assert_array_equal(np.shape(P.derivatives([0, 1])), (n, 2))

    def test_shapes_vectorvalue(self):
        P = KroghInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        assert_array_equal(np.shape(P(0)), (3,))
        assert_array_equal(np.shape(P([0])), (1, 3))
        assert_array_equal(np.shape(P([0, 1])), (2, 3))

    def test_shapes_1d_vectorvalue(self):
        P = KroghInterpolator(self.xs, np.outer(self.ys, [1]))
        assert_array_equal(np.shape(P(0)), (1,))
        assert_array_equal(np.shape(P([0])), (1, 1))
        assert_array_equal(np.shape(P([0, 1])), (2, 1))

    def test_shapes_vectorvalue_derivative(self):
        P = KroghInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n, 3))
        assert_array_equal(np.shape(P.derivatives([0])), (n, 1, 3))
        assert_array_equal(np.shape(P.derivatives([0, 1])), (n, 2, 3))

    def test_wrapper(self):
        P = KroghInterpolator(self.xs, self.ys)
        ki = krogh_interpolate
        assert_almost_equal(P(self.test_xs), ki(self.xs, self.ys, self.test_xs))
        assert_almost_equal(P.derivative(self.test_xs, 2), ki(self.xs, self.ys, self.test_xs, der=2))
        assert_almost_equal(P.derivatives(self.test_xs, 2), ki(self.xs, self.ys, self.test_xs, der=[0, 1]))

    def test_int_inputs(self):
        x = [0, 234, 468, 702, 936, 1170, 1404, 2340, 3744, 6084, 8424, 13104, 60000]
        offset_cdf = np.array([-0.95, -0.86114777, -0.8147762, -0.64072425, -0.48002351, -0.34925329, -0.26503107, -0.13148093, -0.12988833, -0.12979296, -0.12973574, -0.08582937, 0.05])
        f = KroghInterpolator(x, offset_cdf)
        assert_allclose(abs((f(x) - offset_cdf) / f.derivative(x, 1)), 0, atol=1e-10)

    def test_derivatives_complex(self):
        x, y = (np.array([-1, -1, 0, 1, 1]), np.array([1, 1j, 0, -1, 1j]))
        func = KroghInterpolator(x, y)
        cmplx = func.derivatives(0)
        cmplx2 = KroghInterpolator(x, y.real).derivatives(0) + 1j * KroghInterpolator(x, y.imag).derivatives(0)
        assert_allclose(cmplx, cmplx2, atol=1e-15)

    def test_high_degree_warning(self):
        with pytest.warns(UserWarning, match='40 degrees provided,'):
            KroghInterpolator(np.arange(40), np.ones(40))