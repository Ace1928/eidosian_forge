import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
class TestPolynomial(MemoryLeakMixin, TestCase):

    def test_trimseq_basic(self):
        pyfunc = trimseq
        cfunc = njit(trimseq)

        def inputs():
            for i in range(5):
                yield np.array([1] + [0] * i)
        for coefs in inputs():
            self.assertPreciseEqual(pyfunc(coefs), cfunc(coefs))

    def test_trimseq_exception(self):
        cfunc = njit(trimseq)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc('abc')
        self.assertIn('The argument "seq" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as e:
            cfunc(np.arange(10).reshape(5, 2))
        self.assertIn('Coefficient array is not 1-d', str(e.exception))
        with self.assertRaises(TypingError) as e:
            cfunc((1, 2, 3, 0))
        self.assertIn('Unsupported type UniTuple(int64, 4) for argument "seq"', str(e.exception))

    def test_pu_as_series_basic(self):
        pyfunc1 = polyasseries1
        cfunc1 = njit(polyasseries1)
        pyfunc2 = polyasseries2
        cfunc2 = njit(polyasseries2)

        def inputs():
            yield np.arange(4)
            yield np.arange(6).reshape((2, 3))
            yield (1, np.arange(3), np.arange(2, dtype=np.float32))
            yield ([1, 2, 3, 4, 0], [1, 2, 3])
            yield ((0, 0, 0.001, 0, 1e-05, 0, 0), (1, 2, 3, 4, 5, 6, 7))
            yield ((0, 0, 0.001, 0, 1e-05, 0, 0), (1j, 2, 3j, 4j, 5, 6j, 7))
            yield (2, [1.1, 0.0])
            yield ([1, 2, 3, 0],)
            yield ((1, 2, 3, 0),)
            yield (np.array([1, 2, 3, 0]),)
            yield [np.array([1, 2, 3, 0]), np.array([1, 2, 3, 0])]
            yield [np.array([1, 2, 3])]
        for input in inputs():
            self.assertPreciseEqual(pyfunc1(input), cfunc1(input))
            self.assertPreciseEqual(pyfunc2(input, False), cfunc2(input, False))
            self.assertPreciseEqual(pyfunc2(input, True), cfunc2(input, True))

    def test_pu_as_series_exception(self):
        cfunc1 = njit(polyasseries1)
        cfunc2 = njit(polyasseries2)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc1('abc')
        self.assertIn('The argument "alist" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc2('abc', True)
        self.assertIn('The argument "alist" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc2(np.arange(4), 'abc')
        self.assertIn('The argument "trim" must be boolean', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc1(([1, 2, 3], np.arange(16).reshape(4, 4)))
        self.assertIn('Coefficient array is not 1-d', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc1(np.arange(8).reshape((2, 2, 2)))
        self.assertIn('Coefficient array is not 1-d', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc1([np.array([[1, 2, 3], [1, 2, 3]])])
        self.assertIn('Coefficient array is not 1-d', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc1(np.array([[]], dtype=np.float64))
        self.assertIn('Coefficient array is empty', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc1(([1, 2, 3], np.array([], dtype=np.float64), np.array([1, 2, 1])))
        self.assertIn('Coefficient array is empty', str(raises.exception))

    def _test_polyarithm_basic(self, pyfunc, ignore_sign_on_zero=False):
        cfunc = njit(pyfunc)

        def inputs():
            for i in range(5):
                for j in range(5):
                    p1 = np.array([0] * i + [1])
                    p2 = np.array([0] * j + [1])
                    yield (p1, p2)
            yield ([1, 2, 3], [1, 2, 3])
            yield ([1, 2, 3], (1, 2, 3))
            yield ((1, 2, 3), [1, 2, 3])
            yield ([1, 2, 3], 3)
            yield (3, (1, 2, 3))
            yield (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
            yield (np.array([1j, 2j, 3j]), np.array([1.0, 2.0, 3.0]))
            yield (np.array([1, 2, 3]), np.array([1j, 2j, 3j]))
            yield ((1, 2, 3), 3.0)
            yield ((1, 2, 3), 3j)
            yield ((1, 0.001, 3), (1, 2, 3))
        for p1, p2 in inputs():
            self.assertPreciseEqual(pyfunc(p1, p2), cfunc(p1, p2), ignore_sign_on_zero=ignore_sign_on_zero)

    def _test_polyarithm_exception(self, pyfunc):
        cfunc = njit(pyfunc)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc('abc', np.array([1, 2, 3]))
        self.assertIn('The argument "c1" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([1, 2, 3]), 'abc')
        self.assertIn('The argument "c2" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as e:
            cfunc(np.arange(10).reshape(5, 2), np.array([1, 2, 3]))
        self.assertIn('Coefficient array is not 1-d', str(e.exception))
        with self.assertRaises(TypingError) as e:
            cfunc(np.array([1, 2, 3]), np.arange(10).reshape(5, 2))
        self.assertIn('Coefficient array is not 1-d', str(e.exception))

    def test_polyadd_basic(self):
        self._test_polyarithm_basic(polyadd)

    def test_polyadd_exception(self):
        self._test_polyarithm_exception(polyadd)

    def test_polysub_basic(self):
        self._test_polyarithm_basic(polysub, ignore_sign_on_zero=True)

    def test_polysub_exception(self):
        self._test_polyarithm_exception(polysub)

    def test_polymul_basic(self):
        self._test_polyarithm_basic(polymul)

    def test_polymul_exception(self):
        self._test_polyarithm_exception(polymul)

    def test_poly_polydiv_basic(self):
        pyfunc = polydiv
        cfunc = njit(polydiv)
        self._test_polyarithm_basic(polydiv)

        def inputs():
            yield ([2], [2])
            yield ([2, 2], [2])
            for i in range(5):
                for j in range(5):
                    ci = [0] * i + [1, 2]
                    cj = [0] * j + [1, 2]
                    tgt = poly.polyadd(ci, cj)
                    yield (tgt, ci)
            yield (np.array([1, 0, 0, 0, 0, 0, -1]), np.array([1, 0, 0, -1]))
        for c1, c2 in inputs():
            self.assertPreciseEqual(pyfunc(c1, c2), cfunc(c1, c2))

    def test_poly_polydiv_exception(self):
        self._test_polyarithm_exception(polydiv)
        cfunc = njit(polydiv)
        with self.assertRaises(ZeroDivisionError) as _:
            cfunc([1], [0])

    def test_poly_polyval_basic(self):
        pyfunc2 = polyval2
        cfunc2 = njit(polyval2)
        pyfunc3T = polyval3T
        cfunc3T = njit(polyval3T)
        pyfunc3F = polyval3F
        cfunc3F = njit(polyval3F)

        def inputs():
            yield (np.array([], dtype=np.float64), [1])
            yield (1, [1, 2, 3])
            yield (np.arange(4).reshape(2, 2), [1, 2, 3])
            for i in range(5):
                yield (np.linspace(-1, 1), [0] * i + [1])
            yield (np.linspace(-1, 1), [0, -1, 0, 1])
            for i in range(3):
                dims = [2] * i
                x = np.zeros(dims)
                yield (x, [1])
                yield (x, [1, 0])
                yield (x, [1, 0, 0])
            yield (np.array([1, 2]), np.arange(4).reshape(2, 2))
            yield ([1, 2], np.arange(4).reshape(2, 2))
        for x, c in inputs():
            self.assertPreciseEqual(pyfunc2(x, c), cfunc2(x, c))
            self.assertPreciseEqual(pyfunc3T(x, c), cfunc3T(x, c))
            self.assertPreciseEqual(pyfunc3F(x, c), cfunc3F(x, c))

    def test_poly_polyval_exception(self):
        cfunc2 = njit(polyval2)
        cfunc3T = njit(polyval3T)
        cfunc3F = njit(polyval3F)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc2(3, 'abc')
        self.assertIn('The argument "c" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc2('abc', 3)
        self.assertIn('The argument "x" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc2('abc', 'def')
        self.assertIn('The argument "x" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc3T(3, 'abc')
        self.assertIn('The argument "c" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc3T('abc', 3)
        self.assertIn('The argument "x" must be array-like', str(raises.exception))

        @njit
        def polyval3(x, c, tensor):
            res = poly.polyval(x, c, tensor)
            return res
        with self.assertRaises(TypingError) as raises:
            polyval3(3, 3, 'abc')
        self.assertIn('The argument "tensor" must be boolean', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc3F('abc', 'def')
        self.assertIn('The argument "x" must be array-like', str(raises.exception))

    def test_poly_polyint_basic(self):
        pyfunc = polyint
        cfunc = njit(polyint)
        self.assertPreciseEqual(pyfunc([1, 2, 3]), cfunc([1, 2, 3]))
        for i in range(2, 5):
            self.assertPreciseEqual(pyfunc([0], m=i), cfunc([0], m=i))
        for i in range(5):
            pol = [0] * i + [1]
            self.assertPreciseEqual(pyfunc(pol, m=1), pyfunc(pol, m=1))
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                self.assertPreciseEqual(pyfunc(pol, m=j), cfunc(pol, m=j))
        c2 = np.array([[0, 1], [0, 2]])
        self.assertPreciseEqual(pyfunc(c2), cfunc(c2))
        c3 = np.arange(8).reshape((2, 2, 2))
        self.assertPreciseEqual(pyfunc(c3), cfunc(c3))

    def test_poly_polyint_exception(self):
        cfunc = njit(polyint)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc('abc')
        self.assertIn('The argument "c" must be array-like', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([1, 2, 3]), 'abc')
        self.assertIn('The argument "m" must be an integer', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(['a', 'b', 'c'], 1)
        self.assertIn('Input dtype must be scalar.', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(('a', 'b', 'c'), 1)
        self.assertIn('Input dtype must be scalar.', str(raises.exception))

    def test_Polynomial_constructor(self):

        def pyfunc3(c, dom, win):
            p = poly.Polynomial(c, dom, win)
            return p
        cfunc3 = njit(pyfunc3)

        def pyfunc1(c):
            p = poly.Polynomial(c)
            return p
        cfunc1 = njit(pyfunc1)
        list1 = (np.array([0, 1]), np.array([0.0, 1.0]))
        list2 = (np.array([0, 1]), np.array([0.0, 1.0]))
        list3 = (np.array([0, 1]), np.array([0.0, 1.0]))
        for c in list1:
            for dom in list2:
                for win in list3:
                    p1 = pyfunc3(c, dom, win)
                    p2 = cfunc3(c, dom, win)
                    q1 = pyfunc1(c)
                    q2 = cfunc1(c)
                    self.assertPreciseEqual(p1, p2)
                    self.assertPreciseEqual(p1.coef, p2.coef)
                    self.assertPreciseEqual(p1.domain, p2.domain)
                    self.assertPreciseEqual(p1.window, p2.window)
                    self.assertPreciseEqual(q1.coef, q2.coef)
                    self.assertPreciseEqual(q1.domain, q2.domain)
                    self.assertPreciseEqual(q1.window, q2.window)

    def test_Polynomial_exeption(self):

        def pyfunc3(c, dom, win):
            p = poly.Polynomial(c, dom, win)
            return p
        cfunc3 = njit(pyfunc3)
        self.disable_leak_check()
        input2 = np.array([1, 2])
        input3 = np.array([1, 2, 3])
        input2D = np.arange(4).reshape((2, 2))
        with self.assertRaises(ValueError) as raises:
            cfunc3(input2, input3, input2)
        self.assertIn('Domain has wrong number of elements.', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc3(input2, input2, input3)
        self.assertIn('Window has wrong number of elements.', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc3(input2D, input2, input2)
        self.assertIn('Coefficient array is not 1-d', str(raises.exception))