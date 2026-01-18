import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_use_curlybraces_with_initmany(self):

    @njit
    def foo():
        return {1: 2.2, 3: 4.4, 5: 6.6}
    d = foo()
    self.assertEqual(d, {1: 2.2, 3: 4.4, 5: 6.6})