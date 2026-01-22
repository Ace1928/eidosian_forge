import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class _OutSideTest(_ParserState):
    """State for the subunit parser outside of a test context."""

    def lostConnection(self):
        """Connection lost."""

    def startTest(self, offset, line):
        """A test start command received."""
        self.parser._state = self.parser._in_test
        test_name = line[offset:-1].decode('utf8')
        self.parser._current_test = RemotedTestCase(test_name)
        self.parser.current_test_description = test_name
        self.parser.client.startTest(self.parser._current_test)
        self.parser.subunitLineReceived(line)