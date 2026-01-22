import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
class SomeTest(object):

    def _test(self, f, foo):
        test.assertIs(Foo, original_foo)
        test.assertIs(Foo.f, f)
        test.assertEqual(Foo.g, 3)
        test.assertIs(Foo.foo, foo)
        test.assertTrue(is_instance(f, MagicMock))
        test.assertTrue(is_instance(foo, MagicMock))

    def test_two(self, f, foo):
        self._test(f, foo)

    def test_one(self, f, foo):
        self._test(f, foo)