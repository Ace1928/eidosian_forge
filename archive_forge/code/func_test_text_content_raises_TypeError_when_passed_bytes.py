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
def test_text_content_raises_TypeError_when_passed_bytes(self):
    data = _b('Some Bytes')
    self.assertRaises(TypeError, text_content, data)