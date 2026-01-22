from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
def _get_2_to_2by2_numpy(se):
    args = x, y = se.symbols('x y')
    exprs = np.array([[x + y + 1.0, x * y], [x / y, x ** y]])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X + Y + 1.0)) < 1e-15
        assert abs(A[0, 1] - X * Y) < 1e-15
        assert abs(A[1, 0] - X / Y) < 1e-15
        assert abs(A[1, 1] - X ** Y) < 1e-13
    return (l, check)