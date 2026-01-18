import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def var_propagate4(a, b):
    c = 5 + (a - 1 and b + 1) or (a + 1 and b - 1)
    return c