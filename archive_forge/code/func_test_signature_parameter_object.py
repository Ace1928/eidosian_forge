from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_object(self):
    p = inspect.Parameter('foo', default=10, kind=inspect.Parameter.POSITIONAL_ONLY)
    self.assertEqual(p.name, 'foo')
    self.assertEqual(p.default, 10)
    self.assertIs(p.annotation, p.empty)
    self.assertEqual(p.kind, inspect.Parameter.POSITIONAL_ONLY)
    with self.assertRaisesRegex(ValueError, 'invalid value'):
        inspect.Parameter('foo', default=10, kind='123')
    with self.assertRaisesRegex(ValueError, 'not a valid parameter name'):
        inspect.Parameter('1', kind=inspect.Parameter.VAR_KEYWORD)
    with self.assertRaisesRegex(ValueError, 'non-positional-only parameter'):
        inspect.Parameter(None, kind=inspect.Parameter.VAR_KEYWORD)
    with self.assertRaisesRegex(ValueError, 'cannot have default values'):
        inspect.Parameter('a', default=42, kind=inspect.Parameter.VAR_KEYWORD)
    with self.assertRaisesRegex(ValueError, 'cannot have default values'):
        inspect.Parameter('a', default=42, kind=inspect.Parameter.VAR_POSITIONAL)
    p = inspect.Parameter('a', default=42, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    with self.assertRaisesRegex(ValueError, 'cannot have default values'):
        p.replace(kind=inspect.Parameter.VAR_POSITIONAL)
    self.assertTrue(repr(p).startswith('<Parameter'))