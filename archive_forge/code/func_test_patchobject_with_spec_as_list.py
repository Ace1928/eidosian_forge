import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_with_spec_as_list(self):

    @patch.object(SomeClass, 'class_attribute', spec=['wibble'])
    def test(MockAttribute):
        self.assertEqual(SomeClass.class_attribute, MockAttribute)
        self.assertTrue(is_instance(SomeClass.class_attribute.wibble, MagicMock))
        self.assertRaises(AttributeError, lambda: SomeClass.class_attribute.not_wibble)
    test()