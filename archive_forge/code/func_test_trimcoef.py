import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_trimcoef(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, pu.trimcoef, coef, -1)
    assert_equal(pu.trimcoef(coef), coef[:-1])
    assert_equal(pu.trimcoef(coef, 1), coef[:-3])
    assert_equal(pu.trimcoef(coef, 2), [0])