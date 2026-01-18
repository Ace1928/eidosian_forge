import math
import unittest
from numba import jit
from numba.core import types
from numba.core.errors import TypingError, NumbaTypeError
def test_pyobj_func_npm(self):
    with self.assertRaises(TypingError):
        self.test_pyobj_func(flags=no_pyobj_flags)