import unittest
import numpy as np
from numba import jit
from numba.tests.support import override_config
class TestDisabledJIT(unittest.TestCase):

    def test_decorated_function(self):
        with override_config('DISABLE_JIT', True):

            def method(x):
                return x
            jitted = jit(method)
        self.assertEqual(jitted, method)
        self.assertEqual(10, method(10))
        self.assertEqual(10, jitted(10))

    def test_decorated_function_with_kwargs(self):
        with override_config('DISABLE_JIT', True):

            def method(x):
                return x
            jitted = jit(nopython=True)(method)
        self.assertEqual(jitted, method)
        self.assertEqual(10, method(10))
        self.assertEqual(10, jitted(10))