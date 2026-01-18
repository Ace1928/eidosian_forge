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
def test_progress_accepted_extended(self):
    self.result = ExtendedTestResult()
    self.stream = BytesIO()
    self.protocol = subunit.TestProtocolServer(self.result, stream=self.stream)
    self.protocol.lineReceived(_b('progress: 23'))
    self.protocol.lineReceived(_b('progress: push'))
    self.protocol.lineReceived(_b('progress: -2'))
    self.protocol.lineReceived(_b('progress: pop'))
    self.protocol.lineReceived(_b('progress: +4'))
    self.assertEqual(_b(''), self.stream.getvalue())
    self.assertEqual([('progress', 23, subunit.PROGRESS_SET), ('progress', None, subunit.PROGRESS_PUSH), ('progress', -2, subunit.PROGRESS_CUR), ('progress', None, subunit.PROGRESS_POP), ('progress', 4, subunit.PROGRESS_CUR)], self.result._events)