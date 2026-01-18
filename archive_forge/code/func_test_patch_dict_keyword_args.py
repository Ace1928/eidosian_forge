import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_dict_keyword_args(self):
    original = {'foo': 'bar'}
    copy = original.copy()
    patcher = patch.dict(original, foo=3, bar=4, baz=5)
    patcher.start()
    try:
        self.assertEqual(original, dict(foo=3, bar=4, baz=5))
    finally:
        patcher.stop()
    self.assertEqual(original, copy)