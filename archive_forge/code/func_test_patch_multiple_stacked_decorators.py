import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_stacked_decorators(self):
    original_foo = Foo
    original_f = Foo.f
    original_g = Foo.g

    @patch.multiple(foo_name, f=DEFAULT)
    @patch.multiple(foo_name, foo=DEFAULT)
    @patch(foo_name + '.g')
    def test1(g, **kwargs):
        _test(g, **kwargs)

    @patch.multiple(foo_name, f=DEFAULT)
    @patch(foo_name + '.g')
    @patch.multiple(foo_name, foo=DEFAULT)
    def test2(g, **kwargs):
        _test(g, **kwargs)

    @patch(foo_name + '.g')
    @patch.multiple(foo_name, f=DEFAULT)
    @patch.multiple(foo_name, foo=DEFAULT)
    def test3(g, **kwargs):
        _test(g, **kwargs)

    def _test(g, **kwargs):
        f = kwargs.pop('f')
        foo = kwargs.pop('foo')
        self.assertFalse(kwargs)
        self.assertIs(Foo, original_foo)
        self.assertIs(Foo.f, f)
        self.assertIs(Foo.g, g)
        self.assertIs(Foo.foo, foo)
        self.assertTrue(is_instance(f, MagicMock))
        self.assertTrue(is_instance(g, MagicMock))
        self.assertTrue(is_instance(foo, MagicMock))
    test1()
    test2()
    test3()
    self.assertEqual(Foo.f, original_f)
    self.assertEqual(Foo.g, original_g)