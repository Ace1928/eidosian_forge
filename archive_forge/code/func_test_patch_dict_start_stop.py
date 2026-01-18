import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_dict_start_stop(self):
    d = {'foo': 'bar'}
    original = d.copy()
    patcher = patch.dict(d, [('spam', 'eggs')], clear=True)
    self.assertEqual(d, original)
    patcher.start()
    try:
        self.assertEqual(d, {'spam': 'eggs'})
    finally:
        patcher.stop()
    self.assertEqual(d, original)