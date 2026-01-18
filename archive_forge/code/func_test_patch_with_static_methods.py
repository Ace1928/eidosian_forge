import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_with_static_methods(self):

    class Foo(object):

        @staticmethod
        def woot():
            return sentinel.Static

    @patch.object(Foo, 'woot', staticmethod(lambda: sentinel.Patched))
    def anonymous():
        self.assertEqual(Foo.woot(), sentinel.Patched)
    anonymous()
    self.assertEqual(Foo.woot(), sentinel.Static)