from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyline_zero(self):
    assert_equal(poly.polyline(3, 0), [3])