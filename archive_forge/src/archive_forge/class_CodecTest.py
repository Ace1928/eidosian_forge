from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class CodecTest(TestCase):
    """tests bytes/unicode helpers in passlib.utils"""

    def test_bytes(self):
        """test b() helper, bytes and native str type"""
        if PY3:
            import builtins
            self.assertIs(bytes, builtins.bytes)
        else:
            import __builtin__ as builtins
            self.assertIs(bytes, builtins.str)
        self.assertIsInstance(b'', bytes)
        self.assertIsInstance(b'\x00\xff', bytes)
        if PY3:
            self.assertEqual(b'\x00\xff'.decode('latin-1'), '\x00ÿ')
        else:
            self.assertEqual(b'\x00\xff', '\x00ÿ')

    def test_to_bytes(self):
        """test to_bytes()"""
        from passlib.utils import to_bytes
        self.assertEqual(to_bytes(u('abc')), b'abc')
        self.assertEqual(to_bytes(u('\x00ÿ')), b'\x00\xc3\xbf')
        self.assertEqual(to_bytes(u('\x00ÿ'), 'latin-1'), b'\x00\xff')
        self.assertRaises(ValueError, to_bytes, u('\x00ÿ'), 'ascii')
        self.assertEqual(to_bytes(b'abc'), b'abc')
        self.assertEqual(to_bytes(b'\x00\xff'), b'\x00\xff')
        self.assertEqual(to_bytes(b'\x00\xc3\xbf'), b'\x00\xc3\xbf')
        self.assertEqual(to_bytes(b'\x00\xc3\xbf', 'latin-1'), b'\x00\xc3\xbf')
        self.assertEqual(to_bytes(b'\x00\xc3\xbf', 'latin-1', '', 'utf-8'), b'\x00\xff')
        self.assertRaises(AssertionError, to_bytes, 'abc', None)
        self.assertRaises(TypeError, to_bytes, None)

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

    def test_to_native_str(self):
        """test to_native_str()"""
        from passlib.utils import to_native_str
        self.assertEqual(to_native_str(u('abc'), 'ascii'), 'abc')
        self.assertEqual(to_native_str(b'abc', 'ascii'), 'abc')
        if PY3:
            self.assertEqual(to_native_str(u('à'), 'ascii'), 'à')
            self.assertRaises(UnicodeDecodeError, to_native_str, b'\xc3\xa0', 'ascii')
        else:
            self.assertRaises(UnicodeEncodeError, to_native_str, u('à'), 'ascii')
            self.assertEqual(to_native_str(b'\xc3\xa0', 'ascii'), 'Ã\xa0')
        self.assertEqual(to_native_str(u('à'), 'latin-1'), 'à')
        self.assertEqual(to_native_str(b'\xe0', 'latin-1'), 'à')
        self.assertEqual(to_native_str(u('à'), 'utf-8'), 'à' if PY3 else 'Ã\xa0')
        self.assertEqual(to_native_str(b'\xc3\xa0', 'utf-8'), 'à' if PY3 else 'Ã\xa0')
        self.assertRaises(TypeError, to_native_str, None, 'ascii')

    def test_is_ascii_safe(self):
        """test is_ascii_safe()"""
        from passlib.utils import is_ascii_safe
        self.assertTrue(is_ascii_safe(b'\x00abc\x7f'))
        self.assertTrue(is_ascii_safe(u('\x00abc\x7f')))
        self.assertFalse(is_ascii_safe(b'\x00abc\x80'))
        self.assertFalse(is_ascii_safe(u('\x00abc\x80')))

    def test_is_same_codec(self):
        """test is_same_codec()"""
        from passlib.utils import is_same_codec
        self.assertTrue(is_same_codec(None, None))
        self.assertFalse(is_same_codec(None, 'ascii'))
        self.assertTrue(is_same_codec('ascii', 'ascii'))
        self.assertTrue(is_same_codec('ascii', 'ASCII'))
        self.assertTrue(is_same_codec('utf-8', 'utf-8'))
        self.assertTrue(is_same_codec('utf-8', 'utf8'))
        self.assertTrue(is_same_codec('utf-8', 'UTF_8'))
        self.assertFalse(is_same_codec('ascii', 'utf-8'))