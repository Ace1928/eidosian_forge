import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_new_callable_failure(self):
    original_f = Foo.f
    original_g = Foo.g
    original_foo = Foo.foo

    def crasher():
        raise NameError('crasher')
    patcher = patch.object(Foo, 'f', 1)
    patcher.attribute_name = 'f'
    good = patch.object(Foo, 'g', 1)
    good.attribute_name = 'g'
    bad = patch.object(Foo, 'foo', new_callable=crasher)
    bad.attribute_name = 'foo'
    for additionals in ([good, bad], [bad, good]):
        patcher.additional_patchers = additionals

        @patcher
        def func():
            pass
        self.assertRaises(NameError, func)
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)
        self.assertEqual(Foo.foo, original_foo)