import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_raise_invalid_class_nopython(self):
    msg = 'Encountered unsupported constant type used for exception'
    with self.assertRaises(errors.UnsupportedError) as raises:
        self.check_raise_invalid_class(int, flags=no_pyobj_flags)
    self.assertIn(msg, str(raises.exception))
    with self.assertRaises(errors.UnsupportedError) as raises:
        self.check_raise_invalid_class(1, flags=no_pyobj_flags)
    self.assertIn(msg, str(raises.exception))