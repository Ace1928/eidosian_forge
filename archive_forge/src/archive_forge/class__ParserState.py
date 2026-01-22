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
class _ParserState(object):
    """State for the subunit parser."""

    def __init__(self, parser):
        self.parser = parser
        self._test_sym = (_b('test'), _b('testing'))
        self._colon_sym = _b(':')
        self._error_sym = (_b('error'),)
        self._failure_sym = (_b('failure'),)
        self._progress_sym = (_b('progress'),)
        self._skip_sym = _b('skip')
        self._success_sym = (_b('success'), _b('successful'))
        self._tags_sym = (_b('tags'),)
        self._time_sym = (_b('time'),)
        self._xfail_sym = (_b('xfail'),)
        self._uxsuccess_sym = (_b('uxsuccess'),)
        self._start_simple = _u(' [')
        self._start_multipart = _u(' [ multipart')

    def addError(self, offset, line):
        """An 'error:' directive has been read."""
        self.parser.stdOutLineReceived(line)

    def addExpectedFail(self, offset, line):
        """An 'xfail:' directive has been read."""
        self.parser.stdOutLineReceived(line)

    def addFailure(self, offset, line):
        """A 'failure:' directive has been read."""
        self.parser.stdOutLineReceived(line)

    def addSkip(self, offset, line):
        """A 'skip:' directive has been read."""
        self.parser.stdOutLineReceived(line)

    def addSuccess(self, offset, line):
        """A 'success:' directive has been read."""
        self.parser.stdOutLineReceived(line)

    def lineReceived(self, line):
        """a line has been received."""
        parts = line.split(None, 1)
        if len(parts) == 2 and line.startswith(parts[0]):
            cmd, rest = parts
            offset = len(cmd) + 1
            cmd = cmd.rstrip(self._colon_sym)
            if cmd in self._test_sym:
                self.startTest(offset, line)
            elif cmd in self._error_sym:
                self.addError(offset, line)
            elif cmd in self._failure_sym:
                self.addFailure(offset, line)
            elif cmd in self._progress_sym:
                self.parser._handleProgress(offset, line)
            elif cmd in self._skip_sym:
                self.addSkip(offset, line)
            elif cmd in self._success_sym:
                self.addSuccess(offset, line)
            elif cmd in self._tags_sym:
                self.parser._handleTags(offset, line)
                self.parser.subunitLineReceived(line)
            elif cmd in self._time_sym:
                self.parser._handleTime(offset, line)
                self.parser.subunitLineReceived(line)
            elif cmd in self._xfail_sym:
                self.addExpectedFail(offset, line)
            elif cmd in self._uxsuccess_sym:
                self.addUnexpectedSuccess(offset, line)
            else:
                self.parser.stdOutLineReceived(line)
        else:
            self.parser.stdOutLineReceived(line)

    def lostConnection(self):
        """Connection lost."""
        self.parser._lostConnectionInTest(_u('unknown state of '))

    def startTest(self, offset, line):
        """A test start command received."""
        self.parser.stdOutLineReceived(line)