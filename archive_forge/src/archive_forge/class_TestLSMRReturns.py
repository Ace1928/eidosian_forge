from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
class TestLSMRReturns:

    def setup_method(self):
        self.n = 10
        self.A = lowerBidiagonalMatrix(20, self.n)
        self.xtrue = transpose(arange(self.n, 0, -1))
        self.Afun = aslinearoperator(self.A)
        self.b = self.Afun.matvec(self.xtrue)
        self.x0 = ones(self.n)
        self.x00 = self.x0.copy()
        self.returnValues = lsmr(self.A, self.b)
        self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)

    def test_unchanged_x0(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0
        assert_allclose(self.x00, self.x0)

    def testNormr(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert norm(self.b - self.Afun.matvec(x)) == pytest.approx(normr)

    def testNormar(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x))) == pytest.approx(normar)

    def testNormx(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert norm(x) == pytest.approx(normx)