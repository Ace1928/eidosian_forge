from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_poly2leg(self):
    for i in range(10):
        assert_almost_equal(leg.poly2leg(Llist[i]), [0] * i + [1])