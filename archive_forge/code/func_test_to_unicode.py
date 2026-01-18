from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_to_unicode(self):
    """test to_unicode()"""
    from passlib.utils import to_unicode
    self.assertEqual(to_unicode(u('abc')), u('abc'))
    self.assertEqual(to_unicode(u('\x00ÿ')), u('\x00ÿ'))
    self.assertEqual(to_unicode(u('\x00ÿ'), 'ascii'), u('\x00ÿ'))
    self.assertEqual(to_unicode(b'abc'), u('abc'))
    self.assertEqual(to_unicode(b'\x00\xc3\xbf'), u('\x00ÿ'))
    self.assertEqual(to_unicode(b'\x00\xff', 'latin-1'), u('\x00ÿ'))
    self.assertRaises(ValueError, to_unicode, b'\x00\xff')
    self.assertRaises(AssertionError, to_unicode, 'abc', None)
    self.assertRaises(TypeError, to_unicode, None)