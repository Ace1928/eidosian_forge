import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
class TestExtraMethods:
    """
    Test other methods for manipulating/creating polynomial objects.
    """
    p = poly.Polynomial([1, 2, 3, 0], symbol='z')

    def test_copy(self):
        other = self.p.copy()
        assert_equal(other.symbol, 'z')

    def test_trim(self):
        other = self.p.trim()
        assert_equal(other.symbol, 'z')

    def test_truncate(self):
        other = self.p.truncate(2)
        assert_equal(other.symbol, 'z')

    @pytest.mark.parametrize('kwarg', ({'domain': [-10, 10]}, {'window': [-10, 10]}, {'kind': poly.Chebyshev}))
    def test_convert(self, kwarg):
        other = self.p.convert(**kwarg)
        assert_equal(other.symbol, 'z')

    def test_integ(self):
        other = self.p.integ()
        assert_equal(other.symbol, 'z')

    def test_deriv(self):
        other = self.p.deriv()
        assert_equal(other.symbol, 'z')