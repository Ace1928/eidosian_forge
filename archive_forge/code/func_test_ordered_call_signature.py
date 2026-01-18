import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_ordered_call_signature(self):
    m = Mock()
    m.hello(name='hello', daddy='hero')
    text = "call(name='hello', daddy='hero')"
    self.assertEqual(repr(m.hello.call_args), text)