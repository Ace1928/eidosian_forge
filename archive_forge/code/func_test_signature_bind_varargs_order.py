from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_varargs_order(self):

    def test(*args):
        return args
    self.assertEqual(self.call(test), ())
    self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))