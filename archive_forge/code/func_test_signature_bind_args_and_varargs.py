from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_args_and_varargs(self):

    def test(a, b, c=3, *args):
        return (a, b, c, args)
    self.assertEqual(self.call(test, 1, 2, 3, 4, 5), (1, 2, 3, (4, 5)))
    self.assertEqual(self.call(test, 1, 2), (1, 2, 3, ()))
    self.assertEqual(self.call(test, b=1, a=2), (2, 1, 3, ()))
    self.assertEqual(self.call(test, 1, b=2), (1, 2, 3, ()))
    with self.assertRaisesRegex(TypeError, "multiple values for argument 'c'"):
        self.call(test, 1, 2, 3, c=4)