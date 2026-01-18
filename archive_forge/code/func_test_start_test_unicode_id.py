import datetime
import io
import os
import tempfile
import unittest
from io import BytesIO
from testtools import PlaceHolder, TestCase, TestResult, skipIf
from testtools.compat import _b, _u
from testtools.content import Content, TracebackContent, text_content
from testtools.content_type import ContentType
from testtools.matchers import Contains, Equals, MatchesAny
import iso8601
import subunit
from subunit.tests import (_remote_exception_repr,
def test_start_test_unicode_id(self):
    """Test startTest on a TestProtocolClient."""
    self.protocol.startTest(self.unicode_test)
    expected = _b('test: ') + _u('☃').encode('utf8') + _b('\n')
    self.assertEqual(expected, self.io.getvalue())