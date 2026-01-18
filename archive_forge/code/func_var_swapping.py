import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def var_swapping(a, b, c, d, e):
    a, b = (b, a)
    c, d, e = (e, c, d)
    a, b, c, d = (b, c, d, a)
    return a + b + c + d + e