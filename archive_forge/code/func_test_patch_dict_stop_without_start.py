import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_dict_stop_without_start(self):
    d = {'foo': 'bar'}
    original = d.copy()
    patcher = patch.dict(d, [('spam', 'eggs')], clear=True)
    self.assertFalse(patcher.stop())
    self.assertEqual(d, original)