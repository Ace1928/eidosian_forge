import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_stop_without_start(self):
    patcher = patch(foo_name, 'bar', 3)
    self.assertRaises(RuntimeError, patcher.stop)