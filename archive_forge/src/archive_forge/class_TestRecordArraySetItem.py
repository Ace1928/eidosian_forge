import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
class TestRecordArraySetItem(TestCase):
    """
    Test setitem when index is Literal[str]
    """

    def test_literal_variable(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = set_field1
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0].copy()), jitfunc(arr[0].copy()))

    def test_literal_unroll(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = set_field2
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0].copy()), jitfunc(arr[0].copy()))

    def test_literal_variable_global_tuple(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = set_field3
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0].copy()), jitfunc(arr[0].copy()))

    def test_literal_unroll_global_tuple(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = set_field4
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0].copy()), jitfunc(arr[0].copy()))

    def test_literal_unroll_free_var_tuple(self):
        arr = np.array([1, 2], dtype=recordtype2)
        fs = arr.dtype.names

        def set_field(rec):
            for f in literal_unroll(fs):
                rec[f] = 10
            return rec
        jitfunc = njit(set_field)
        self.assertEqual(set_field(arr[0].copy()), jitfunc(arr[0].copy()))

    def test_error_w_invalid_field(self):
        arr = np.array([1, 2], dtype=recordtype3)
        jitfunc = njit(set_field1)
        with self.assertRaises(TypingError) as raises:
            jitfunc(arr[0])
        self.assertIn("Field 'f' was not found in record with fields ('first', 'second')", str(raises.exception))