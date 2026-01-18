from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_replace(self):
    p = inspect.Parameter('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY)
    self.assertIsNot(p, p.replace())
    self.assertEqual(p, p.replace())
    p2 = p.replace(annotation=1)
    self.assertEqual(p2.annotation, 1)
    p2 = p2.replace(annotation=p2.empty)
    self.assertEqual(p, p2)
    p2 = p2.replace(name='bar')
    self.assertEqual(p2.name, 'bar')
    self.assertNotEqual(p2, p)
    with self.assertRaisesRegex(ValueError, 'not a valid parameter name'):
        p2 = p2.replace(name=p2.empty)
    p2 = p2.replace(name='foo', default=None)
    self.assertIs(p2.default, None)
    self.assertNotEqual(p2, p)
    p2 = p2.replace(name='foo', default=p2.empty)
    self.assertIs(p2.default, p2.empty)
    p2 = p2.replace(default=42, kind=p2.POSITIONAL_OR_KEYWORD)
    self.assertEqual(p2.kind, p2.POSITIONAL_OR_KEYWORD)
    self.assertNotEqual(p2, p)
    with self.assertRaisesRegex(ValueError, 'invalid value for'):
        p2 = p2.replace(kind=p2.empty)
    p2 = p2.replace(kind=p2.KEYWORD_ONLY)
    self.assertEqual(p2, p)