from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import unittest, TestCase
def test_unicode_fname(self):
    fname = u'foಠ'
    argtypes = (types.int32, types.float32)
    name = default_mangler(fname, argtypes)
    self.assertIsInstance(name, str)
    unichar = fname[2]
    enc = ''.join(('_{:02x}'.format(c) for c in unichar.encode('utf8')))
    text = 'fo' + enc
    expect = '_Z{}{}if'.format(len(text), text)
    self.assertEqual(name, expect)
    self.assertRegex(name, '^_Z[a-zA-Z0-9_\\$]+$')