import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_use_curlybraces(self):

    @njit
    def foo():
        d = {}
        d[1] = 2
        return d
    d = foo()
    self.assertEqual(d, {1: 2})