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
def test_passing_array_ctypes_voidptr(self):
    """
        Test the ".ctypes" attribute of an array can be passed
        as a "void *" parameter.
        """
    self.check_array_ctypes(use_c_vsquare)