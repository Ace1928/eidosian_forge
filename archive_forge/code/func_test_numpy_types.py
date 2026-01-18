import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def test_numpy_types(self):
    """
        Test explicit casting to Numpy number types.
        """

    def tp_factory(tp_name):
        return getattr(np, tp_name)
    self.check_number_types(tp_factory)