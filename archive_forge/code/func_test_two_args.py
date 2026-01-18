from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import unittest, TestCase
def test_two_args(self):
    fname = 'foo'
    argtypes = (types.int32, types.float32)
    name = default_mangler(fname, argtypes)
    self.assertEqual(name, '_Z3fooif')