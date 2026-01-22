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
class TestCTypesTypes(TestCase):

    def _conversion_tests(self, check):
        check(c_double, types.float64)
        check(c_int, types.intc)
        check(c_uint16, types.uint16)
        check(c_size_t, types.size_t)
        check(c_ssize_t, types.ssize_t)
        check(c_void_p, types.voidptr)
        check(POINTER(c_float), types.CPointer(types.float32))
        check(POINTER(POINTER(c_float)), types.CPointer(types.CPointer(types.float32)))
        check(None, types.void)

    def test_from_ctypes(self):
        """
        Test converting a ctypes type to a Numba type.
        """

        def check(cty, ty):
            got = ctypes_utils.from_ctypes(cty)
            self.assertEqual(got, ty)
        self._conversion_tests(check)
        with self.assertRaises(TypeError) as raises:
            ctypes_utils.from_ctypes(c_wchar_p)
        self.assertIn('Unsupported ctypes type', str(raises.exception))

    def test_to_ctypes(self):
        """
        Test converting a Numba type to a ctypes type.
        """

        def check(cty, ty):
            got = ctypes_utils.to_ctypes(ty)
            self.assertEqual(got, cty)
        self._conversion_tests(check)
        with self.assertRaises(TypeError) as raises:
            ctypes_utils.to_ctypes(types.ellipsis)
        self.assertIn("Cannot convert Numba type '...' to ctypes type", str(raises.exception))