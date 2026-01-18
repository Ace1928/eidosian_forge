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
def test_top_frame_is_skipped_when_no_stack_is_specified(self):
    actual = StacktraceContent().as_text()
    self.assertNotIn('testtools/content.py', actual)