import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
class TestBarycentric:

    def setup_method(self):
        self.true_poly = np.polynomial.Polynomial([-4, 5, 1, 3, -2])
        self.test_xs = np.linspace(-1, 1, 100)
        self.xs = np.linspace(-1, 1, 5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_allclose(P(self.test_xs), self.true_poly(self.test_xs))

    def test_scalar(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_allclose(P(7), self.true_poly(7))
        assert_allclose(P(np.array(7)), self.true_poly(np.array(7)))

    def test_derivatives(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        D = P.derivatives(self.test_xs)
        for i in range(D.shape[0]):
            assert_allclose(self.true_poly.deriv(i)(self.test_xs), D[i])

    def test_low_derivatives(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        D = P.derivatives(self.test_xs, len(self.xs) + 2)
        for i in range(D.shape[0]):
            assert_allclose(self.true_poly.deriv(i)(self.test_xs), D[i], atol=1e-12)

    def test_derivative(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        m = 10
        r = P.derivatives(self.test_xs, m)
        for i in range(m):
            assert_allclose(P.derivative(self.test_xs, i), r[i])

    def test_high_derivative(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        for i in range(len(self.xs), 5 * len(self.xs)):
            assert_allclose(P.derivative(self.test_xs, i), np.zeros(len(self.test_xs)))

    def test_ndim_derivatives(self):
        poly1 = self.true_poly
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
        P = BarycentricInterpolator(self.xs, ys, axis=0)
        D = P.derivatives(self.test_xs)
        for i in range(D.shape[0]):
            assert_allclose(D[i], np.stack((poly1.deriv(i)(self.test_xs), poly2.deriv(i)(self.test_xs), poly3.deriv(i)(self.test_xs)), axis=-1), atol=1e-12)

    def test_ndim_derivative(self):
        poly1 = self.true_poly
        poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
        poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
        ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
        P = BarycentricInterpolator(self.xs, ys, axis=0)
        for i in range(P.n):
            assert_allclose(P.derivative(self.test_xs, i), np.stack((poly1.deriv(i)(self.test_xs), poly2.deriv(i)(self.test_xs), poly3.deriv(i)(self.test_xs)), axis=-1), atol=1e-12)

    def test_delayed(self):
        P = BarycentricInterpolator(self.xs)
        P.set_yi(self.ys)
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_append(self):
        P = BarycentricInterpolator(self.xs[:3], self.ys[:3])
        P.add_xi(self.xs[3:], self.ys[3:])
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_vector(self):
        xs = [0, 1, 2]
        ys = np.array([[0, 1], [1, 0], [2, 1]])
        BI = BarycentricInterpolator
        P = BI(xs, ys)
        Pi = [BI(xs, ys[:, i]) for i in range(ys.shape[1])]
        test_xs = np.linspace(-1, 3, 100)
        assert_almost_equal(P(test_xs), np.asarray([p(test_xs) for p in Pi]).T)

    def test_shapes_scalarvalue(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_array_equal(np.shape(P(0)), ())
        assert_array_equal(np.shape(P(np.array(0))), ())
        assert_array_equal(np.shape(P([0])), (1,))
        assert_array_equal(np.shape(P([0, 1])), (2,))

    def test_shapes_scalarvalue_derivative(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n,))
        assert_array_equal(np.shape(P.derivatives(np.array(0))), (n,))
        assert_array_equal(np.shape(P.derivatives([0])), (n, 1))
        assert_array_equal(np.shape(P.derivatives([0, 1])), (n, 2))

    def test_shapes_vectorvalue(self):
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        assert_array_equal(np.shape(P(0)), (3,))
        assert_array_equal(np.shape(P([0])), (1, 3))
        assert_array_equal(np.shape(P([0, 1])), (2, 3))

    def test_shapes_1d_vectorvalue(self):
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, [1]))
        assert_array_equal(np.shape(P(0)), (1,))
        assert_array_equal(np.shape(P([0])), (1, 1))
        assert_array_equal(np.shape(P([0, 1])), (2, 1))

    def test_shapes_vectorvalue_derivative(self):
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n, 3))
        assert_array_equal(np.shape(P.derivatives([0])), (n, 1, 3))
        assert_array_equal(np.shape(P.derivatives([0, 1])), (n, 2, 3))

    def test_wrapper(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        bi = barycentric_interpolate
        assert_allclose(P(self.test_xs), bi(self.xs, self.ys, self.test_xs))
        assert_allclose(P.derivative(self.test_xs, 2), bi(self.xs, self.ys, self.test_xs, der=2))
        assert_allclose(P.derivatives(self.test_xs, 2), bi(self.xs, self.ys, self.test_xs, der=[0, 1]))

    def test_int_input(self):
        x = 1000 * np.arange(1, 11)
        y = np.arange(1, 11)
        value = barycentric_interpolate(x, y, 1000 * 9.5)
        assert_almost_equal(value, 9.5)

    def test_large_chebyshev(self):
        n = 1100
        j = np.arange(n + 1).astype(np.float64)
        x = np.cos(j * np.pi / n)
        w = (-1) ** j
        w[0] *= 0.5
        w[-1] *= 0.5
        P = BarycentricInterpolator(x)
        factor = P.wi[0]
        assert_almost_equal(P.wi / (2 * factor), w)

    def test_warning(self):
        P = BarycentricInterpolator([0, 1], [1, 2])
        with np.errstate(divide='raise'):
            yi = P(P.xi)
        assert_almost_equal(yi, P.yi.ravel())

    def test_repeated_node(self):
        xis = np.array([0.1, 0.5, 0.9, 0.5])
        ys = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match='Interpolation points xi must be distinct.'):
            BarycentricInterpolator(xis, ys)