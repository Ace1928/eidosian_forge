import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_pow_op_usecase(self):
    args = [(2, 3), (2.0, 3), (2, 3.0), (2j, 3j)]
    for x, y in args:
        argtys = (typeof(x), typeof(y))
        cfunc = jit(argtys, **no_pyobj_flags)(pow_op_usecase)
        r = cfunc(x, y)
        self.assertPreciseEqual(r, pow_op_usecase(x, y))