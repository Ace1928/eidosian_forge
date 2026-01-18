from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_saslprep(self):
    """test saslprep() unicode normalizer"""
    self.require_stringprep()
    from passlib.utils import saslprep as sp
    self.assertRaises(TypeError, sp, None)
    self.assertRaises(TypeError, sp, 1)
    self.assertRaises(TypeError, sp, b'')
    self.assertEqual(sp(u('')), u(''))
    self.assertEqual(sp(u('\xad')), u(''))
    self.assertEqual(sp(u('$\xad$\u200d$')), u('$$$'))
    self.assertEqual(sp(u('$ $\xa0$\u3000$')), u('$ $ $ $'))
    self.assertEqual(sp(u('à')), u('à'))
    self.assertEqual(sp(u('à')), u('à'))
    self.assertRaises(ValueError, sp, u('\x00'))
    self.assertRaises(ValueError, sp, u('\x7f'))
    self.assertRaises(ValueError, sp, u('\u180e'))
    self.assertRaises(ValueError, sp, u('\ufff9'))
    self.assertRaises(ValueError, sp, u('\ue000'))
    self.assertRaises(ValueError, sp, u('\ufdd0'))
    self.assertRaises(ValueError, sp, u('\ud800'))
    self.assertRaises(ValueError, sp, u('�'))
    self.assertRaises(ValueError, sp, u('⿰'))
    self.assertRaises(ValueError, sp, u('\u200e'))
    self.assertRaises(ValueError, sp, u('\u206f'))
    self.assertRaises(ValueError, sp, u('ऀ'))
    self.assertRaises(ValueError, sp, u('\ufff8'))
    self.assertRaises(ValueError, sp, u('\U000e0001'))
    self.assertRaises(ValueError, sp, u('ا1'))
    self.assertEqual(sp(u('ا')), u('ا'))
    self.assertEqual(sp(u('اب')), u('اب'))
    self.assertEqual(sp(u('ا1ب')), u('ا1ب'))
    self.assertRaises(ValueError, sp, u('اAب'))
    self.assertRaises(ValueError, sp, u('xاz'))
    self.assertEqual(sp(u('xAz')), u('xAz'))
    self.assertEqual(sp(u('I\xadX')), u('IX'))
    self.assertEqual(sp(u('user')), u('user'))
    self.assertEqual(sp(u('USER')), u('USER'))
    self.assertEqual(sp(u('ª')), u('a'))
    self.assertEqual(sp(u('Ⅸ')), u('IX'))
    self.assertRaises(ValueError, sp, u('\x07'))
    self.assertRaises(ValueError, sp, u('ا1'))
    self.assertRaises(ValueError, sp, u('ا1'))
    self.assertEqual(sp(u('ا1ب')), u('ا1ب'))