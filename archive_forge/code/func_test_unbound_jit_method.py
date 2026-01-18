import unittest
import numpy as np
from numba import jit
from numba.tests.support import override_config
def test_unbound_jit_method(self):

    class Something(object):

        def __init__(self, x0):
            self.x0 = x0

        @jit(forceobj=True)
        def method(self):
            return self.x0
    something = Something(3)
    self.assertEqual(Something.method(something), 3)