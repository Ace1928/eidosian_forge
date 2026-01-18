import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_pow_too_large(self):
    assert_raises(ValueError, pu._pow, (), [1, 2, 3], 5, 4)