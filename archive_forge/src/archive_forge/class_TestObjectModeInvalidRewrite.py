import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
class TestObjectModeInvalidRewrite(TestCase):
    """
    Tests to ensure that rewrite passes didn't affect objmode lowering.
    """

    def _ensure_objmode(self, disp):
        self.assertTrue(disp.signatures)
        self.assertFalse(disp.nopython_signatures)
        return disp

    def test_static_raise_in_objmode_fallback(self):
        """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2159
        """

        def test0(n):
            return n

        def test1(n):
            if n == 0:
                raise ValueError()
            return test0(n)
        compiled = jit(forceobj=True)(test1)
        self.assertEqual(test1(10), compiled(10))
        self._ensure_objmode(compiled)

    def test_static_setitem_in_objmode_fallback(self):
        """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2169
        """

        def test0(n):
            return n

        def test(a1, a2):
            a1 = np.asarray(a1)
            a2[0] = 1
            return test0(a1.sum() + a2.sum())
        compiled = jit(forceobj=True)(test)
        args = (np.array([3]), np.array([4]))
        self.assertEqual(test(*args), compiled(*args))
        self._ensure_objmode(compiled)

    def test_dynamic_func_objmode(self):
        """
        Test issue https://github.com/numba/numba/issues/3355
        """
        func_text = 'def func():\n'
        func_text += '    np.array([1,2,3])\n'
        loc_vars = {}
        custom_globals = {'np': np}
        exec(func_text, custom_globals, loc_vars)
        func = loc_vars['func']
        jitted = jit(forceobj=True)(func)
        jitted()