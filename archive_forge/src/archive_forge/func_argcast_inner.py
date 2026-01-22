from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
@jit(nopython=True)
def argcast_inner(a, b):
    if b:
        a = int64(0)
    return a