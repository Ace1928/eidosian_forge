import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_with_none(self):

    class Something(object):
        attribute = sentinel.Original

    @patch.object(Something, 'attribute', None)
    def test():
        self.assertIsNone(Something.attribute, 'unpatched')
    test()
    self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')