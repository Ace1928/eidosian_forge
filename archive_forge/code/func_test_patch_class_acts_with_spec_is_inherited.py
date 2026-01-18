import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_class_acts_with_spec_is_inherited(self):

    @patch('%s.SomeClass' % __name__, spec=True)
    def test(MockSomeClass):
        self.assertTrue(is_instance(MockSomeClass, MagicMock))
        instance = MockSomeClass()
        self.assertNotCallable(instance)
        instance.wibble
        self.assertRaises(AttributeError, lambda: instance.not_wibble)
    test()