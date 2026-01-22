import unittest
import numpy as np
from numba import jit
from numba.tests.support import override_config
class TestJITMethod(unittest.TestCase):

    def test_bound_jit_method_with_loop_lift(self):

        class Something(object):

            def __init__(self, x0):
                self.x0 = x0

            @jit(forceobj=True)
            def method(self, x):
                a = np.empty(shape=5, dtype=np.float32)
                x0 = self.x0
                for i in range(a.shape[0]):
                    a[i] = x0 * x
                return a
        something = Something(3)
        np.testing.assert_array_equal(something.method(5), np.array([15, 15, 15, 15, 15], dtype=np.float32))
        [cres] = something.method.overloads.values()
        jitloop = cres.lifted[0]
        [loopcres] = jitloop.overloads.values()
        self.assertTrue(loopcres.fndesc.native)

    def test_unbound_jit_method(self):

        class Something(object):

            def __init__(self, x0):
                self.x0 = x0

            @jit(forceobj=True)
            def method(self):
                return self.x0
        something = Something(3)
        self.assertEqual(Something.method(something), 3)