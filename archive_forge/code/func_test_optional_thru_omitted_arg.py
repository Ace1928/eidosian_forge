import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_optional_thru_omitted_arg(self):
    """
        Issue 1868
        """

    def pyfunc(x=None):
        if x is None:
            x = 1
        return x
    cfunc = njit(pyfunc)
    self.assertEqual(pyfunc(), cfunc())
    self.assertEqual(pyfunc(3), cfunc(3))