from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
def test_star_call_objmode(self):
    self.test_star_call(objmode=True)