from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_immutability(self):
    p = inspect.Parameter(None, kind=inspect.Parameter.POSITIONAL_ONLY)
    with self.assertRaises(AttributeError):
        p.foo = 'bar'
    with self.assertRaises(AttributeError):
        p.kind = 123