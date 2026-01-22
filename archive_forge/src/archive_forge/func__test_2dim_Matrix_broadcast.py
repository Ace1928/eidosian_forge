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
def _test_2dim_Matrix_broadcast():
    L, check = _get_1_to_2by3_matrix()
    inp = range(1, 5)
    out = L(inp)
    for i in range(len(inp)):
        check(out[i, ...], (inp[i],))