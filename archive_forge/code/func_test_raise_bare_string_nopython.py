import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_raise_bare_string_nopython(self):

    @njit
    def foo():
        raise 'illegal'
    msg = 'Directly raising a string constant as an exception is not supported'
    with self.assertRaises(errors.UnsupportedError) as raises:
        foo()
    self.assertIn(msg, str(raises.exception))