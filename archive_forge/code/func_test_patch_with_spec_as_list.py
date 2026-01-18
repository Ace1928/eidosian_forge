import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_with_spec_as_list(self):

    @patch('%s.SomeClass' % __name__, spec=['wibble'])
    def test(MockSomeClass):
        self.assertEqual(SomeClass, MockSomeClass)
        self.assertTrue(is_instance(SomeClass.wibble, MagicMock))
        self.assertRaises(AttributeError, lambda: SomeClass.not_wibble)
    test()