import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class MockMedium(medium.SmartClientMedium):
    """A mock medium that can be used to test _SmartClient.

    It can be given a series of requests to expect (and responses it should
    return for them).  It can also be told when the client is expected to
    disconnect a medium.  Expectations must be satisfied in the order they are
    given, or else an AssertionError will be raised.

    Typical use looks like::

        medium = MockMedium()
        medium.expect_request(...)
        medium.expect_request(...)
        medium.expect_request(...)
    """

    def __init__(self):
        super().__init__('dummy base')
        self._mock_request = _MockMediumRequest(self)
        self._expected_events = []

    def expect_request(self, request_bytes, response_bytes, allow_partial_read=False):
        """Expect 'request_bytes' to be sent, and reply with 'response_bytes'.

        No assumption is made about how many times accept_bytes should be
        called to send the request.  Similarly, no assumption is made about how
        many times read_bytes/read_line are called by protocol code to read a
        response.  e.g.::

            request.accept_bytes(b'ab')
            request.accept_bytes(b'cd')
            request.finished_writing()

        and::

            request.accept_bytes(b'abcd')
            request.finished_writing()

        Will both satisfy ``medium.expect_request('abcd', ...)``.  Thus tests
        using this should not break due to irrelevant changes in protocol
        implementations.

        :param allow_partial_read: if True, no assertion is raised if a
            response is not fully read.  Setting this is useful when the client
            is expected to disconnect without needing to read the complete
            response.  Default is False.
        """
        self._expected_events.append(('send request', request_bytes))
        if allow_partial_read:
            self._expected_events.append(('read response (partial)', response_bytes))
        else:
            self._expected_events.append(('read response', response_bytes))

    def expect_disconnect(self):
        """Expect the client to call ``medium.disconnect()``."""
        self._expected_events.append('disconnect')

    def _assertEvent(self, observed_event):
        """Raise AssertionError unless observed_event matches the next expected
        event.

        :seealso: expect_request
        :seealso: expect_disconnect
        """
        try:
            expected_event = self._expected_events.pop(0)
        except IndexError:
            raise AssertionError('Mock medium observed event %r, but no more events expected' % (observed_event,))
        if expected_event[0] == 'read response (partial)':
            if observed_event[0] != 'read response':
                raise AssertionError('Mock medium observed event %r, but expected event %r' % (observed_event, expected_event))
        elif observed_event != expected_event:
            raise AssertionError('Mock medium observed event %r, but expected event %r' % (observed_event, expected_event))
        if self._expected_events:
            next_event = self._expected_events[0]
            if next_event[0].startswith('read response'):
                self._mock_request._response = next_event[1]

    def get_request(self):
        return self._mock_request

    def disconnect(self):
        if self._mock_request._read_bytes:
            self._assertEvent(('read response', self._mock_request._read_bytes))
            self._mock_request._read_bytes = b''
        self._assertEvent('disconnect')