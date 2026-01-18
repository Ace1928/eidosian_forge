import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_stop_idempotent(self):
    patcher = patch(foo_name, 'bar', 3)
    patcher.start()
    patcher.stop()
    self.assertIsNone(patcher.stop())