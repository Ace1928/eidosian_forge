from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def test_passing_array_ctypes_voidptr_pass_ptr(self):
    """
        Test the ".ctypes" attribute of an array can be passed
        as a pointer parameter of the right type.
        """
    cfunc = self.check_array_ctypes(use_c_vcube)
    with self.assertRaises(errors.TypingError) as raises:
        cfunc(np.float32([0.0]))
    self.assertIn('No implementation of function ExternalFunctionPointer', str(raises.exception))