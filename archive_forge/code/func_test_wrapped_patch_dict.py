import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_wrapped_patch_dict(self):
    decorated = patch.dict('sys.modules')(function)
    self.assertIs(decorated.__wrapped__, function)