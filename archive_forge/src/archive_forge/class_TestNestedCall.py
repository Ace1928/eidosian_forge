from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
class TestNestedCall(TestCase):

    def compile_func(self, pyfunc, objmode=False):

        def check(*args, **kwargs):
            expected = pyfunc(*args, **kwargs)
            result = f(*args, **kwargs)
            self.assertPreciseEqual(result, expected)
        flags = dict(forceobj=True) if objmode else dict(nopython=True)
        f = jit(**flags)(pyfunc)
        return (f, check)

    def test_boolean_return(self):

        @jit(nopython=True)
        def inner(x):
            return not x

        @jit(nopython=True)
        def outer(x):
            if inner(x):
                return True
            else:
                return False
        self.assertFalse(outer(True))
        self.assertTrue(outer(False))

    def test_named_args(self, objmode=False):
        """
        Test a nested function call with named (keyword) arguments.
        """
        cfunc, check = self.compile_func(f, objmode)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_named_args_objmode(self):
        self.test_named_args(objmode=True)

    def test_default_args(self, objmode=False):
        """
        Test a nested function call using default argument values.
        """
        cfunc, check = self.compile_func(g, objmode)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_default_args_objmode(self):
        self.test_default_args(objmode=True)

    def test_star_args(self):
        """
        Test a nested function call to a function with *args in its signature.
        """
        cfunc, check = self.compile_func(star)
        check(1, 2, 3)

    def test_star_call(self, objmode=False):
        """
        Test a function call with a *args.
        """
        cfunc, check = self.compile_func(star_call, objmode)
        check(1, (2,), (3,))

    def test_star_call_objmode(self):
        self.test_star_call(objmode=True)

    def test_argcast(self):
        """
        Issue #1488: implicitly casting an argument variable should not
        break nested calls.
        """
        cfunc, check = self.compile_func(argcast)
        check(1, 0)
        check(1, 1)

    def test_call_generated(self):
        """
        Test a nested function call to a generated jit function.
        """
        cfunc = jit(nopython=True)(call_generated)
        self.assertPreciseEqual(cfunc(1, 2), (-4, 2))
        self.assertPreciseEqual(cfunc(1j, 2), (1j + 5, 2))