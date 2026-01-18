import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def test_numba_carray(self):
    """
        Test Numba-compiled carray() against pure Python carray()
        """
    self.check_numba_carray_farray(carray_usecase, carray_dtype_usecase)