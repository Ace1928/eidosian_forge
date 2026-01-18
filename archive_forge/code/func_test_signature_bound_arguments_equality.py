from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bound_arguments_equality(self):

    def foo(a):
        pass
    ba = inspect.signature(foo).bind(1)
    self.assertEqual(ba, ba)
    ba2 = inspect.signature(foo).bind(1)
    self.assertEqual(ba, ba2)
    ba3 = inspect.signature(foo).bind(2)
    self.assertNotEqual(ba, ba3)
    ba3.arguments['a'] = 1
    self.assertEqual(ba, ba3)

    def bar(b):
        pass
    ba4 = inspect.signature(bar).bind(1)
    self.assertNotEqual(ba, ba4)