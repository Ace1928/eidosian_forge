import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
def test_scalarmul(self):
    out = self.p * 10
    assert_equal(out.symbol, 'z')