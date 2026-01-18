import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_try_raise(self):

    @njit
    def raise_(a):
        raise ValueError(a)

    @njit
    def try_raise(a):
        try:
            raise_(a)
        except Exception:
            pass
        return a + 1
    self.assertEqual(try_raise.py_func(3), try_raise(3))