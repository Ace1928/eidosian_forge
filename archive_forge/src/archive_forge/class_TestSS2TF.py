from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class TestSS2TF:

    def check_matrix_shapes(self, p, q, r):
        ss2tf(np.zeros((p, p)), np.zeros((p, q)), np.zeros((r, p)), np.zeros((r, q)), 0)

    def test_shapes(self):
        for p, q, r in [(3, 3, 3), (1, 3, 3), (1, 1, 1)]:
            self.check_matrix_shapes(p, q, r)

    def test_basic(self):
        b = np.array([1.0, 3.0, 5.0])
        a = np.array([1.0, 2.0, 3.0])
        A, B, C, D = tf2ss(b, a)
        assert_allclose(A, [[-2, -3], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2]], rtol=1e-13)
        assert_allclose(D, [[1]], rtol=1e-14)
        bb, aa = ss2tf(A, B, C, D)
        assert_allclose(bb[0], b, rtol=1e-13)
        assert_allclose(aa, a, rtol=1e-13)

    def test_zero_order_round_trip(self):
        tf = (2, 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0]], rtol=1e-13)
        assert_allclose(D, [[2]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)
        tf = ([[5], [2]], 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0], [0]], rtol=1e-13)
        assert_allclose(D, [[5], [2]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[5, 0], [2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)

    def test_simo_round_trip(self):
        tf = ([[1, 2], [1, 1]], [1, 2])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2]], rtol=1e-13)
        assert_allclose(B, [[1]], rtol=1e-13)
        assert_allclose(C, [[0], [-1]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 2], [1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 2], rtol=1e-13)
        tf = ([[1, 0, 1], [1, 1, 1]], [1, 1, 1])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-1, -1], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[-1, 0], [0, 0]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 0, 1], [1, 1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 1, 1], rtol=1e-13)
        tf = ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2, -3, -4], [1, 0, 0], [0, 1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2, 3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(D, [[0], [0]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, 2, 3], [0, 1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 2, 3, 4], rtol=1e-13)
        tf = (np.array([1, [2, 3]], dtype=object), [1, 6])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6]], rtol=1e-31)
        assert_allclose(B, [[1]], rtol=1e-31)
        assert_allclose(C, [[1], [-9]], rtol=1e-31)
        assert_allclose(D, [[0], [2]], rtol=1e-31)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1], [2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6], rtol=1e-13)
        tf = (np.array([[1, -3], [1, 2, 3]], dtype=object), [1, 6, 5])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6, -5], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, -3], [-4, -2]], rtol=1e-13)
        assert_allclose(D, [[0], [1]], rtol=1e-13)
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, -3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6, 5], rtol=1e-13)

    def test_all_int_arrays(self):
        A = [[0, 1, 0], [0, 0, 1], [-3, -4, -2]]
        B = [[0], [0], [1]]
        C = [[5, 1, 0]]
        D = [[0]]
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0.0, 0.0, 1.0, 5.0]], rtol=1e-13, atol=1e-14)
        assert_allclose(den, [1.0, 2.0, 4.0, 3.0], rtol=1e-13)

    def test_multioutput(self):
        A = np.array([[-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 2.0, 0.0], [-4.0, 0.0, 3.0, 0.0], [-8.0, 8.0, 0.0, 4.0]])
        B = np.array([[0.3], [0.0], [7.0], [0.0]])
        C = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [8.0, 8.0, 0.0, 0.0]])
        D = np.array([[0.0], [0.0], [1.0]])
        b_all, a = ss2tf(A, B, C, D)
        b0, a0 = ss2tf(A, B, C[0], D[0])
        b1, a1 = ss2tf(A, B, C[1], D[1])
        b2, a2 = ss2tf(A, B, C[2], D[2])
        assert_allclose(a0, a, rtol=1e-13)
        assert_allclose(a1, a, rtol=1e-13)
        assert_allclose(a2, a, rtol=1e-13)
        assert_allclose(b_all, np.vstack((b0, b1, b2)), rtol=1e-13, atol=1e-14)