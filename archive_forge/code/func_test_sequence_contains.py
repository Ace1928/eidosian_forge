import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_sequence_contains(self):
    """
        Test handling of the `in` comparison
        """

    @jit(forceobj=True)
    def foo(x, y):
        return x in y
    self.assertTrue(foo(1, [0, 1]))
    self.assertTrue(foo(0, [0, 1]))
    self.assertFalse(foo(2, [0, 1]))
    with self.assertRaises(TypeError) as raises:
        foo(None, None)
    self.assertIn('is not iterable', str(raises.exception))