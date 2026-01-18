import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_create_mocks_different_order(self):
    original_f = Foo.f
    original_g = Foo.g
    patcher = patch.object(Foo, 'f', 3)
    patcher.attribute_name = 'f'
    other = patch.object(Foo, 'g', DEFAULT)
    other.attribute_name = 'g'
    patcher.additional_patchers = [other]

    @patcher
    def test(g):
        self.assertIs(Foo.g, g)
        self.assertEqual(Foo.f, 3)
    test()
    self.assertEqual(Foo.f, original_f)
    self.assertEqual(Foo.g, original_g)