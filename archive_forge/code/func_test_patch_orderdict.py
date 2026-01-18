import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_orderdict(self):
    foo = OrderedDict()
    foo['a'] = object()
    foo['b'] = 'python'
    original = foo.copy()
    update_values = list(zip('cdefghijklmnopqrstuvwxyz', range(26)))
    patched_values = list(foo.items()) + update_values
    with patch.dict(foo, OrderedDict(update_values)):
        self.assertEqual(list(foo.items()), patched_values)
    self.assertEqual(foo, original)
    with patch.dict(foo, update_values):
        self.assertEqual(list(foo.items()), patched_values)
    self.assertEqual(foo, original)