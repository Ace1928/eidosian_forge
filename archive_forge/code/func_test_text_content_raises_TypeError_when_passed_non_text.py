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
def test_text_content_raises_TypeError_when_passed_non_text(self):
    bad_values = (None, list(), dict(), 42, 1.23)
    for value in bad_values:
        self.assertThat(lambda: text_content(value), raises(TypeError("text_content must be given text, not '%s'." % type(value).__name__)))