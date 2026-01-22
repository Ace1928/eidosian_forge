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
def _get_1_to_2by3_matrix(Mtx=se.DenseMatrix):
    x = se.symbols('x')
    args = (x,)
    exprs = Mtx(2, 3, [x + 1, x + 2, x + 3, 1 / x, 1 / (x * x), 1 / x ** 3.0])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, = inp
        assert abs(A[0, 0] - (X + 1)) < 1e-15
        assert abs(A[0, 1] - (X + 2)) < 1e-15
        assert abs(A[0, 2] - (X + 3)) < 1e-15
        assert abs(A[1, 0] - 1 / X) < 1e-15
        assert abs(A[1, 1] - 1 / (X * X)) < 1e-15
        assert abs(A[1, 2] - 1 / X ** 3.0) < 1e-15
    return (L, check)