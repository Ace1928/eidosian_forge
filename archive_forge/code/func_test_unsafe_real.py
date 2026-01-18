import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_unsafe_real():
    L, check = _get_2_to_2by2_list()
    inp = np.array([13.0, 17.0])
    out = np.empty(4)
    L.unsafe_real(inp, out)
    check(out.reshape((2, 2)), inp)