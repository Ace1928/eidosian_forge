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
def test_unsafe_complex():
    L, check = _get_2_to_2by2_list(real=False)
    assert not L.real
    inp = np.array([13 + 11j, 7 + 4j], dtype=np.complex128)
    out = np.empty(4, dtype=np.complex128)
    L.unsafe_complex(inp, out)
    check(out.reshape((2, 2)), inp)