from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyadd(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            tgt = np.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = poly.polyadd([0] * i + [1], [0] * j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)