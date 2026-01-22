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
class TestStacktraceContent(TestCase):

    def test___init___sets_ivars(self):
        content = StacktraceContent()
        content_type = ContentType('text', 'x-traceback', {'language': 'python', 'charset': 'utf8'})
        self.assertEqual(content_type, content.content_type)

    def test_prefix_is_used(self):
        prefix = self.getUniqueString()
        actual = StacktraceContent(prefix_content=prefix).as_text()
        self.assertTrue(actual.startswith(prefix))

    def test_postfix_is_used(self):
        postfix = self.getUniqueString()
        actual = StacktraceContent(postfix_content=postfix).as_text()
        self.assertTrue(actual.endswith(postfix))

    def test_top_frame_is_skipped_when_no_stack_is_specified(self):
        actual = StacktraceContent().as_text()
        self.assertNotIn('testtools/content.py', actual)