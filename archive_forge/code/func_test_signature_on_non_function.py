from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_on_non_function(self):
    with self.assertRaisesRegex(TypeError, 'is not a callable object'):
        inspect.signature(42)
    with self.assertRaisesRegex(TypeError, 'is not a Python function'):
        inspect.Signature.from_function(42)