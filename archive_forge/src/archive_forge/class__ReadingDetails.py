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
class _ReadingDetails(_ParserState):
    """Common logic for readin state details."""

    def endDetails(self):
        """The end of a details section has been reached."""
        self.parser._state = self.parser._outside_test
        self.parser.current_test_description = None
        self._report_outcome()
        self.parser.client.stopTest(self.parser._current_test)

    def lineReceived(self, line):
        """a line has been received."""
        self.details_parser.lineReceived(line)
        self.parser.subunitLineReceived(line)

    def lostConnection(self):
        """Connection lost."""
        self.parser._lostConnectionInTest(_u('%s report of ') % self._outcome_label())

    def _outcome_label(self):
        """The label to describe this outcome."""
        raise NotImplementedError(self._outcome_label)

    def set_simple(self):
        """Start a simple details parser."""
        self.details_parser = details.SimpleDetailsParser(self)

    def set_multipart(self):
        """Start a multipart details parser."""
        self.details_parser = details.MultipartDetailsParser(self)