from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_immutability(self):

    def test(a):
        pass
    sig = inspect.signature(test)
    with self.assertRaises(AttributeError):
        sig.foo = 'bar'
    if sys.version_info[:2] < (3, 3):
        return
    with self.assertRaises(TypeError):
        sig.parameters['a'] = None