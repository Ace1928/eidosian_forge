import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_trimseq(self):
    for i in range(5):
        tgt = [1]
        res = pu.trimseq([1] + [0] * 5)
        assert_equal(res, tgt)