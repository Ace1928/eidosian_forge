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
class TestStackLinesContent(TestCase):

    def _get_stack_line_and_expected_output(self):
        stack_lines = [('/path/to/file', 42, 'some_function', 'print("Hello World")')]
        expected = '  File "/path/to/file", line 42, in some_function\n    print("Hello World")\n'
        return (stack_lines, expected)

    def test_single_stack_line(self):
        stack_lines, expected = self._get_stack_line_and_expected_output()
        actual = StackLinesContent(stack_lines).as_text()
        self.assertEqual(expected, actual)

    def test_prefix_content(self):
        stack_lines, expected = self._get_stack_line_and_expected_output()
        prefix = self.getUniqueString() + '\n'
        content = StackLinesContent(stack_lines, prefix_content=prefix)
        actual = content.as_text()
        expected = prefix + expected
        self.assertEqual(expected, actual)

    def test_postfix_content(self):
        stack_lines, expected = self._get_stack_line_and_expected_output()
        postfix = '\n' + self.getUniqueString()
        content = StackLinesContent(stack_lines, postfix_content=postfix)
        actual = content.as_text()
        expected = expected + postfix
        self.assertEqual(expected, actual)

    def test___init___sets_content_type(self):
        stack_lines, expected = self._get_stack_line_and_expected_output()
        content = StackLinesContent(stack_lines)
        expected_content_type = ContentType('text', 'x-traceback', {'language': 'python', 'charset': 'utf8'})
        self.assertEqual(expected_content_type, content.content_type)