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
def test_stdout_passthrough(self):
    """Lines received which cannot be interpreted as any protocol action
        should be passed through to sys.stdout.
        """
    bytes = _b('randombytes\n')
    self.protocol.lineReceived(bytes)
    self.assertEqual(self.stdout.getvalue(), bytes)