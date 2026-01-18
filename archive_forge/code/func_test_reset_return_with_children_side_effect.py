import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_reset_return_with_children_side_effect(self):
    m = MagicMock(f=MagicMock(side_effect=[2, 3]))
    self.assertNotEqual(m.f.side_effect, None)
    m.reset_mock(side_effect=True)
    self.assertEqual(m.f.side_effect, None)