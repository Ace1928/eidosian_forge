from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_empty(self):

    def test():
        return 42
    self.assertEqual(self.call(test), 42)
    with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
        self.call(test, 1)
    with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
        self.call(test, 1, spam=10)
    with self.assertRaisesRegex(TypeError, 'too many keyword arguments'):
        self.call(test, spam=1)