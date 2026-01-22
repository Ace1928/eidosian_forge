from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
class TestParameterObject(unittest.TestCase):

    def test_signature_parameter_kinds(self):
        P = inspect.Parameter
        self.assertTrue(P.POSITIONAL_ONLY < P.POSITIONAL_OR_KEYWORD < P.VAR_POSITIONAL < P.KEYWORD_ONLY < P.VAR_KEYWORD)
        self.assertEqual(str(P.POSITIONAL_ONLY), 'POSITIONAL_ONLY')
        self.assertTrue('POSITIONAL_ONLY' in repr(P.POSITIONAL_ONLY))

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

    def test_signature_parameter_equality(self):
        P = inspect.Parameter
        p = P('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(p, p)
        self.assertNotEqual(p, 42)
        self.assertEqual(p, P('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY))

    def test_signature_parameter_unhashable(self):
        p = inspect.Parameter('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaisesRegex(TypeError, 'unhashable type'):
            hash(p)

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

    def test_signature_parameter_positional_only(self):
        p = inspect.Parameter(None, kind=inspect.Parameter.POSITIONAL_ONLY)
        self.assertEqual(str(p), '<>')
        p = p.replace(name='1')
        self.assertEqual(str(p), '<1>')

    def test_signature_parameter_immutability(self):
        p = inspect.Parameter(None, kind=inspect.Parameter.POSITIONAL_ONLY)
        with self.assertRaises(AttributeError):
            p.foo = 'bar'
        with self.assertRaises(AttributeError):
            p.kind = 123