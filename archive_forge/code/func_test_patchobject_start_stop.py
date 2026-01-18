import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_start_stop(self):
    original = something
    patcher = patch.object(PTModule, 'something', 'foo')
    self.assertIs(something, original)
    replaced = patcher.start()
    try:
        self.assertEqual(replaced, 'foo')
        self.assertIs(something, replaced)
    finally:
        patcher.stop()
    self.assertIs(something, original)