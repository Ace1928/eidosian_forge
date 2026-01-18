import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_class_decorator(self):

    class Something(object):
        attribute = sentinel.Original

    class Foo(object):

        def test_method(other_self, mock_something):
            self.assertEqual(PTModule.something, mock_something, 'unpatched')

        def not_test_method(other_self):
            self.assertEqual(PTModule.something, sentinel.Something, 'non-test method patched')
    Foo = patch('%s.something' % __name__)(Foo)
    f = Foo()
    f.test_method()
    f.not_test_method()
    self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')
    self.assertEqual(PTModule.something, sentinel.Something, 'patch not restored')