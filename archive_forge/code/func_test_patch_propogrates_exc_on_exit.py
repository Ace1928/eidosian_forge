import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_propogrates_exc_on_exit(self):

    class holder:
        exc_info = (None, None, None)

    class custom_patch(_patch):

        def __exit__(self, etype=None, val=None, tb=None):
            _patch.__exit__(self, etype, val, tb)
            holder.exc_info = (etype, val, tb)
        stop = __exit__

    def with_custom_patch(target):
        getter, attribute = _get_target(target)
        return custom_patch(getter, attribute, DEFAULT, None, False, None, None, None, {})

    @with_custom_patch('squizz.squozz')
    def test(mock):
        raise RuntimeError
    self.assertRaises(RuntimeError, test)
    self.assertIs(holder.exc_info[0], RuntimeError)
    self.assertIsNotNone(holder.exc_info[1], 'exception value not propgated')
    self.assertIsNotNone(holder.exc_info[2], 'exception traceback not propgated')