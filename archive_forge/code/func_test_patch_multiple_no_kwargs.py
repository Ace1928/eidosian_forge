import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_no_kwargs(self):
    self.assertRaises(ValueError, patch.multiple, foo_name)
    self.assertRaises(ValueError, patch.multiple, Foo)