from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
class TestLSMR:

    def setup_method(self):
        self.n = 10
        self.m = 10

    def assertCompatibleSystem(self, A, xtrue):
        Afun = aslinearoperator(A)
        b = Afun.matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testIdentityACase1(self):
        A = eye(self.n)
        xtrue = zeros((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase2(self):
        A = eye(self.n)
        xtrue = ones((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase3(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1))
        self.assertCompatibleSystem(A, xtrue)

    def testBidiagonalA(self):
        A = lowerBidiagonalMatrix(20, self.n)
        xtrue = transpose(arange(self.n, 0, -1))
        self.assertCompatibleSystem(A, xtrue)

    def testScalarB(self):
        A = array([[1.0, 2.0]])
        b = 3.0
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b) == pytest.approx(0)

    def testComplexX(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexX0(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1))
        b = aslinearoperator(A).matvec(xtrue)
        x0 = zeros(self.n, dtype=complex)
        x = lsmr(A, b, x0=x0)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testComplexA(self):
        A = 4 * eye(self.n) + 1j * ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1).astype(complex))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexB(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        b = aslinearoperator(A).matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-05)

    def testColumnB(self):
        A = eye(self.n)
        b = ones((self.n, 1))
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)

    def testInitialization(self):
        x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
        assert_allclose(norm(b - G @ x_ref), normr_ref, atol=1e-06)
        x0 = zeros(b.shape)
        x = lsmr(G, b, x0=x0)[0]
        assert_allclose(x, x_ref)
        x0 = lsmr(G, b, maxiter=1)[0]
        x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
        assert_allclose(norm(b - G @ x), normr, atol=1e-06)
        assert itn - itn_ref in (0, 1)
        assert normr < normr_ref * (1 + 1e-06)