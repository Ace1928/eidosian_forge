from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_equality(self):
    P = inspect.Parameter
    p = P('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(p, p)
    self.assertNotEqual(p, 42)
    self.assertEqual(p, P('foo', default=42, kind=inspect.Parameter.KEYWORD_ONLY))