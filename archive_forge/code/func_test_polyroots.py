from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyroots(self):
    assert_almost_equal(poly.polyroots([1]), [])
    assert_almost_equal(poly.polyroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = np.linspace(-1, 1, i)
        res = poly.polyroots(poly.polyfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))