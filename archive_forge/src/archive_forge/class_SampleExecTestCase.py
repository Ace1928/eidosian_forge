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
class SampleExecTestCase(subunit.ExecTestCase):

    def test_sample_method(self):
        """sample-script.py"""

    def test_sample_method_args(self):
        """sample-script.py foo"""