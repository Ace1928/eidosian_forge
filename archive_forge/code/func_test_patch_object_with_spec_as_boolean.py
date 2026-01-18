import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_object_with_spec_as_boolean(self):

    @patch.object(PTModule, 'SomeClass', spec=True)
    def test(MockSomeClass):
        self.assertEqual(SomeClass, MockSomeClass)
        MockSomeClass.wibble
        self.assertRaises(AttributeError, lambda: MockSomeClass.not_wibble)
    test()