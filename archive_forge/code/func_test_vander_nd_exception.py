import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_vander_nd_exception(self):
    assert_raises(ValueError, pu._vander_nd, (), (1, 2, 3), [90])
    assert_raises(ValueError, pu._vander_nd, (), (), [90.65])
    assert_raises(ValueError, pu._vander_nd, (), (), [])