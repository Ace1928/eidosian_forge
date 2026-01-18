import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_reset_return_with_children(self):
    m = MagicMock(f=MagicMock(return_value=1))
    self.assertEqual(m.f(), 1)
    m.reset_mock(return_value=True)
    self.assertNotEqual(m.f(), 1)