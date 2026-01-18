import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_failure(self):
    original_f = Foo.f
    original_g = Foo.g
    patcher = patch.object(Foo, 'f', 1)
    patcher.attribute_name = 'f'
    good = patch.object(Foo, 'g', 1)
    good.attribute_name = 'g'
    bad = patch.object(Foo, 'missing', 1)
    bad.attribute_name = 'missing'
    for additionals in ([good, bad], [bad, good]):
        patcher.additional_patchers = additionals

        @patcher
        def func():
            pass
        self.assertRaises(AttributeError, func)
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)