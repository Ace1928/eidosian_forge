import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@patch('mock.patch.TEST_PREFIX', 'bar')
def test_patch_dict_test_prefix(self):

    class Foo(object):

        def bar_one(self):
            return dict(the_dict)

        def bar_two(self):
            return dict(the_dict)

        def test_one(self):
            return dict(the_dict)

        def test_two(self):
            return dict(the_dict)
    the_dict = {'key': 'original'}
    Foo = patch.dict(the_dict, key='changed')(Foo)
    foo = Foo()
    self.assertEqual(foo.bar_one(), {'key': 'changed'})
    self.assertEqual(foo.bar_two(), {'key': 'changed'})
    self.assertEqual(foo.test_one(), {'key': 'original'})
    self.assertEqual(foo.test_two(), {'key': 'original'})