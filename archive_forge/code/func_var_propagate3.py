import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def var_propagate3(a, b):
    c = 5 + (a > b and a or b)
    return c