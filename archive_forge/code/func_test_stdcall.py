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
@unittest.skipUnless(is_windows, 'Windows-specific test')
def test_stdcall(self):
    cfunc = njit((types.uintc,))(use_c_sleep)
    cfunc(1)