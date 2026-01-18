import unittest
from itertools import product
from numba import types, njit, typed, errors
from numba.tests.support import TestCase
def test_standard_getitem_on_type(self):
    with self.assertRaises(errors.TypingError) as raises:

        @njit
        def foo(not_static):
            types.float64[not_static]
        foo(slice(None, None, 1))
    msg = ('No implementation', 'getitem(class(float64), slice<a:b>)')
    excstr = str(raises.exception)
    for m in msg:
        self.assertIn(m, excstr)