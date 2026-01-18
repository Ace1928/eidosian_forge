from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polydomain(self):
    assert_equal(poly.polydomain, [-1, 1])