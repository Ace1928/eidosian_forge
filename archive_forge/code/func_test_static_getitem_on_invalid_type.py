import unittest
from itertools import product
from numba import types, njit, typed, errors
from numba.tests.support import TestCase
def test_static_getitem_on_invalid_type(self):
    types.void[:]
    with self.assertRaises(errors.TypingError) as raises:

        @njit
        def foo():
            types.void[:]
        foo()
    msg = ('No implementation', 'getitem(typeref[none], slice<a:b>)')
    excstr = str(raises.exception)
    for m in msg:
        self.assertIn(m, excstr)