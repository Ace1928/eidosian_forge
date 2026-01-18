import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
def test_single_stack_line(self):
    stack_lines, expected = self._get_stack_line_and_expected_output()
    actual = StackLinesContent(stack_lines).as_text()
    self.assertEqual(expected, actual)