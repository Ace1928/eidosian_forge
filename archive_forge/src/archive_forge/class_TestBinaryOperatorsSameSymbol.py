import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
@pytest.mark.parametrize('rhs', (poly.Polynomial([4, 5, 6], symbol='z'), array([4, 5, 6])))
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_add(self, rhs):
        out = self.p + rhs
        assert_equal(out.symbol, 'z')

    def test_sub(self, rhs):
        out = self.p - rhs
        assert_equal(out.symbol, 'z')

    def test_polymul(self, rhs):
        out = self.p * rhs
        assert_equal(out.symbol, 'z')

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):
            assert_equal(out.symbol, 'z')

    def test_radd(self, rhs):
        out = rhs + self.p
        assert_equal(out.symbol, 'z')

    def test_rsub(self, rhs):
        out = rhs - self.p
        assert_equal(out.symbol, 'z')

    def test_rmul(self, rhs):
        out = rhs * self.p
        assert_equal(out.symbol, 'z')

    def test_rdivmod(self, rhs):
        for out in divmod(rhs, self.p):
            assert_equal(out.symbol, 'z')