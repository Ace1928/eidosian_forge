import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_
@pytest.mark.parametrize(('bad_input', 'exception'), (('', ValueError), ('3', ValueError), (None, TypeError), (1, TypeError)))
def test_symbol_bad_input(self, bad_input, exception):
    with pytest.raises(exception):
        p = poly.Polynomial(self.c, symbol=bad_input)