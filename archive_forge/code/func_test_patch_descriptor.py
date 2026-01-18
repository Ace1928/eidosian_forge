import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@unittest.expectedFailure
def test_patch_descriptor(self):

    class Nothing(object):
        foo = None

    class Something(object):
        foo = {}

        @patch.object(Nothing, 'foo', 2)
        @classmethod
        def klass(cls):
            self.assertIs(cls, Something)

        @patch.object(Nothing, 'foo', 2)
        @staticmethod
        def static(arg):
            return arg

        @patch.dict(foo)
        @classmethod
        def klass_dict(cls):
            self.assertIs(cls, Something)

        @patch.dict(foo)
        @staticmethod
        def static_dict(arg):
            return arg
    self.assertEqual(Something.static('f00'), 'f00')
    Something.klass()
    self.assertEqual(Something.static_dict('f00'), 'f00')
    Something.klass_dict()
    something = Something()
    self.assertEqual(something.static('f00'), 'f00')
    something.klass()
    self.assertEqual(something.static_dict('f00'), 'f00')
    something.klass_dict()