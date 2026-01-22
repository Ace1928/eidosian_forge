import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
class TestFuncLifetime(TestCase):
    """
    Test the lifetime of compiled function objects and their dependencies.
    """

    def get_impl(self, dispatcher):
        """
        Get the single implementation (a C function object) of a dispatcher.
        """
        self.assertEqual(len(dispatcher.overloads), 1)
        cres = list(dispatcher.overloads.values())[0]
        return cres.entry_point

    def check_local_func_lifetime(self, **jitargs):

        def f(x):
            return x + 1
        c_f = jit('int32(int32)', **jitargs)(f)
        self.assertPreciseEqual(c_f(1), 2)
        cfunc = self.get_impl(c_f)
        refs = [weakref.ref(obj) for obj in (f, c_f, cfunc.__self__)]
        obj = f = c_f = cfunc = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_local_func_lifetime(self):
        self.check_local_func_lifetime(forceobj=True)

    def test_local_func_lifetime_npm(self):
        self.check_local_func_lifetime(nopython=True)

    def check_global_func_lifetime(self, **jitargs):
        c_f = jit(**jitargs)(global_usecase1)
        self.assertPreciseEqual(c_f(1), 2)
        cfunc = self.get_impl(c_f)
        wr = weakref.ref(c_f)
        refs = [weakref.ref(obj) for obj in (c_f, cfunc.__self__)]
        obj = c_f = cfunc = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_global_func_lifetime(self):
        self.check_global_func_lifetime(forceobj=True)

    def test_global_func_lifetime_npm(self):
        self.check_global_func_lifetime(nopython=True)

    def check_global_obj_lifetime(self, **jitargs):
        global global_obj
        global_obj = Dummy()
        c_f = jit(**jitargs)(global_usecase2)
        self.assertPreciseEqual(c_f(), 6)
        refs = [weakref.ref(obj) for obj in (c_f, global_obj)]
        obj = c_f = global_obj = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_global_obj_lifetime(self):
        self.check_global_obj_lifetime(forceobj=True)

    def check_inner_function_lifetime(self, **jitargs):
        """
        When a jitted function calls into another jitted function, check
        that everything is collected as desired.
        """

        def mult_10(a):
            return a * 10
        c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
        c_mult_10.disable_compile()

        def do_math(x):
            return c_mult_10(x + 4)
        c_do_math = jit('intp(intp)', **jitargs)(do_math)
        c_do_math.disable_compile()
        self.assertEqual(c_do_math(1), 50)
        wrs = [weakref.ref(obj) for obj in (mult_10, c_mult_10, do_math, c_do_math, self.get_impl(c_mult_10).__self__, self.get_impl(c_do_math).__self__)]
        obj = mult_10 = c_mult_10 = do_math = c_do_math = None
        gc.collect()
        self.assertEqual([w() for w in wrs], [None] * len(wrs))

    def test_inner_function_lifetime(self):
        self.check_inner_function_lifetime(forceobj=True)

    def test_inner_function_lifetime_npm(self):
        self.check_inner_function_lifetime(nopython=True)