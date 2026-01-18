import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_dict(self):
    foo = {'initial': object(), 'other': 'something'}
    original = foo.copy()

    @patch.dict(foo)
    def test():
        foo['a'] = 3
        del foo['initial']
        foo['other'] = 'something else'
    test()
    self.assertEqual(foo, original)

    @patch.dict(foo, {'a': 'b'})
    def test():
        self.assertEqual(len(foo), 3)
        self.assertEqual(foo['a'], 'b')
    test()
    self.assertEqual(foo, original)

    @patch.dict(foo, [('a', 'b')])
    def test():
        self.assertEqual(len(foo), 3)
        self.assertEqual(foo['a'], 'b')
    test()
    self.assertEqual(foo, original)