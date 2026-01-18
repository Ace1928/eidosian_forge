import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_usercode_update_use_d2(self):
    """
        Tests an example using a regular update is
        not modified by the optimization.
        """

    def check_before(x):
        pass

    def check_after(x):
        pass
    checked_before = False
    checked_after = False

    @overload(check_before, prefer_literal=True)
    def ol_check_before(d):
        nonlocal checked_before
        if not checked_before:
            checked_before = True
            a = {'a': 1, 'b': 2, 'c': 3}
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, a)
        return lambda d: None

    @overload(check_after, prefer_literal=True)
    def ol_check_after(d):
        nonlocal checked_after
        if not checked_after:
            checked_after = True
            self.assertTrue(isinstance(d, types.DictType))
            self.assertTrue(d.initial_value is None)
        return lambda d: None

    def const_dict_func():
        """
            Dictionary update between two constant
            dictionaries. This verifies d2 doesn't
            get incorrectly removed.
            """
        d1 = {'a': 1, 'b': 2, 'c': 3}
        d2 = {'d': 4, 'e': 4}
        check_before(d1)
        d1.update(d2)
        check_after(d1)
        if len(d1) > 4:
            return d2
        return d1
    py_func = const_dict_func
    cfunc = njit()(const_dict_func)
    a = py_func()
    b = cfunc()
    self.assertEqual(a, b)