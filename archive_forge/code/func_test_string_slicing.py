import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_string_slicing(self):
    pyfunc = usecases.string_slicing
    cfunc = jit((types.pyobject,) * 3, forceobj=True)(pyfunc)
    test_str = '123456'
    self.assertEqual(pyfunc(test_str, 0, 3), cfunc(test_str, 0, 3))
    self.assertEqual(pyfunc(test_str, 1, 5), cfunc(test_str, 1, 5))
    self.assertEqual(pyfunc(test_str, 2, 3), cfunc(test_str, 2, 3))