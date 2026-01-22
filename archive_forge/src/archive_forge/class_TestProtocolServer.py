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
class TestProtocolServer(object):
    """A parser for subunit.

    :ivar tags: The current tags associated with the protocol stream.
    """

    def __init__(self, client, stream=None, forward_stream=None):
        """Create a TestProtocolServer instance.

        :param client: An object meeting the unittest.TestResult protocol.
        :param stream: The stream that lines received which are not part of the
            subunit protocol should be written to. This allows custom handling
            of mixed protocols. By default, sys.stdout will be used for
            convenience. It should accept bytes to its write() method.
        :param forward_stream: A stream to forward subunit lines to. This
            allows a filter to forward the entire stream while still parsing
            and acting on it. By default forward_stream is set to
            DiscardStream() and no forwarding happens.
        """
        self.client = ExtendedToOriginalDecorator(client)
        if stream is None:
            stream = sys.stdout.buffer
        self._stream = stream
        self._forward_stream = forward_stream or DiscardStream()
        self._in_test = _InTest(self)
        self._outside_test = _OutSideTest(self)
        self._reading_error_details = _ReadingErrorDetails(self)
        self._reading_failure_details = _ReadingFailureDetails(self)
        self._reading_skip_details = _ReadingSkipDetails(self)
        self._reading_success_details = _ReadingSuccessDetails(self)
        self._reading_xfail_details = _ReadingExpectedFailureDetails(self)
        self._reading_uxsuccess_details = _ReadingUnexpectedSuccessDetails(self)
        self._state = self._outside_test
        self._plusminus = _b('+-')
        self._push_sym = _b('push')
        self._pop_sym = _b('pop')

    def _handleProgress(self, offset, line):
        """Process a progress directive."""
        line = line[offset:].strip()
        if line[0] in self._plusminus:
            whence = PROGRESS_CUR
            delta = int(line)
        elif line == self._push_sym:
            whence = PROGRESS_PUSH
            delta = None
        elif line == self._pop_sym:
            whence = PROGRESS_POP
            delta = None
        else:
            whence = PROGRESS_SET
            delta = int(line)
        self.client.progress(delta, whence)

    def _handleTags(self, offset, line):
        """Process a tags command."""
        tags = line[offset:].decode('utf8').split()
        new_tags, gone_tags = tags_to_new_gone(tags)
        self.client.tags(new_tags, gone_tags)

    def _handleTime(self, offset, line):
        try:
            event_time = iso8601.parse_date(line[offset:-1].decode())
        except TypeError:
            raise TypeError(_u('Failed to parse %r, got %r') % (line, sys.exc_info()[1]))
        self.client.time(event_time)

    def lineReceived(self, line):
        """Call the appropriate local method for the received line."""
        self._state.lineReceived(line)

    def _lostConnectionInTest(self, state_string):
        error_string = _u("lost connection during %stest '%s'") % (state_string, self.current_test_description)
        self.client.addError(self._current_test, RemoteError(error_string))
        self.client.stopTest(self._current_test)

    def lostConnection(self):
        """The input connection has finished."""
        self._state.lostConnection()

    def readFrom(self, pipe):
        """Blocking convenience API to parse an entire stream.

        :param pipe: A file-like object supporting __iter__.
        :return: None.
        """
        for line in pipe:
            self.lineReceived(line)
        self.lostConnection()

    def _startTest(self, offset, line):
        """Internal call to change state machine. Override startTest()."""
        self._state.startTest(offset, line)

    def subunitLineReceived(self, line):
        self._forward_stream.write(line)

    def stdOutLineReceived(self, line):
        self._stream.write(line)