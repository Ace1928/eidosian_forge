import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_reset_mock_does_not_raise_on_attr_deletion(self):
    mock = Mock()
    mock.child = True
    del mock.child
    mock.reset_mock()
    self.assertFalse(hasattr(mock, 'child'))