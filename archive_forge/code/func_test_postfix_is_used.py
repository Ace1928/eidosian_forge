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
def test_postfix_is_used(self):
    postfix = self.getUniqueString()
    actual = StacktraceContent(postfix_content=postfix).as_text()
    self.assertTrue(actual.endswith(postfix))