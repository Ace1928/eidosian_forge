import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_class_decorator(self):

    class Something(object):
        attribute = sentinel.Original

    class Foo(object):

        def test_method(other_self):
            self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')

        def not_test_method(other_self):
            self.assertEqual(Something.attribute, sentinel.Original, 'non-test method patched')
    Foo = patch.object(Something, 'attribute', sentinel.Patched)(Foo)
    f = Foo()
    f.test_method()
    f.not_test_method()
    self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')