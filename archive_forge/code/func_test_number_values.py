import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
def test_number_values(self):
    """
        Test special.typeof() with scalar number values.
        """
    self.check_number_values(typeof)
    self.assertEqual(typeof(1), types.intp)
    self.assertEqual(typeof(-1), types.intp)
    self.assertEqual(typeof(2 ** 40), types.int64)
    self.assertEqual(typeof(2 ** 63), types.uint64)
    self.assertEqual(typeof(2 ** 63 - 1), types.int64)
    self.assertEqual(typeof(-2 ** 63), types.int64)