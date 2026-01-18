from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_sre_character_literals(self):
    for i in [0, 8, 16, 32, 64, 127, 128, 255]:
        self.assertEqual(bool(regex.match('\\%03o' % i, chr(i))), True)
        self.assertEqual(bool(regex.match('\\%03o0' % i, chr(i) + '0')), True)
        self.assertEqual(bool(regex.match('\\%03o8' % i, chr(i) + '8')), True)
        self.assertEqual(bool(regex.match('\\x%02x' % i, chr(i))), True)
        self.assertEqual(bool(regex.match('\\x%02x0' % i, chr(i) + '0')), True)
        self.assertEqual(bool(regex.match('\\x%02xz' % i, chr(i) + 'z')), True)
    self.assertRaisesRegex(regex.error, self.INVALID_GROUP_REF, lambda: regex.match('\\911', ''))