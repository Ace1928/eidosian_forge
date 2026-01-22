import math
import unittest
from numba import jit
from numba.core import types
from numba.core.errors import TypingError, NumbaTypeError
class TestReturnValues(unittest.TestCase):

    def test_nopython_func(self, flags=enable_pyobj_flags):
        pyfunc = get_nopython_func
        cfunc = jit((), **flags)(pyfunc)
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, abs)
        else:
            self.fail('Unexpected successful compilation.')

    def test_nopython_func_npm(self):
        with self.assertRaises(NumbaTypeError):
            self.test_nopython_func(flags=no_pyobj_flags)

    def test_pyobj_func(self, flags=enable_pyobj_flags):
        pyfunc = get_pyobj_func
        cfunc = jit((), **flags)(pyfunc)
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, open)
        else:
            self.fail('Unexpected successful compilation.')

    def test_pyobj_func_npm(self):
        with self.assertRaises(TypingError):
            self.test_pyobj_func(flags=no_pyobj_flags)

    def test_module_func(self, flags=enable_pyobj_flags):
        pyfunc = get_module_func
        cfunc = jit((), **flags)(pyfunc)
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, math.floor)
        else:
            self.fail('Unexpected successful compilation.')

    def test_module_func_npm(self):
        with self.assertRaises(NumbaTypeError):
            self.test_module_func(flags=no_pyobj_flags)