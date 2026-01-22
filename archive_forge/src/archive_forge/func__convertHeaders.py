import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
def _convertHeaders(self, headers):
    """
        This method converts the HTTP/2 header set into something that looks
        like HTTP/1.1. In particular, it strips the 'special' headers and adds
        a Host: header.

        @param headers: The HTTP/2 header set.
        @type headers: A L{list} of L{tuple}s of header name and header value,
            both as L{bytes}.
        """
    gotLength = False
    for header in headers:
        if not header[0].startswith(b':'):
            gotLength = _addHeaderToRequest(self._request, header) or gotLength
        elif header[0] == b':method':
            self.command = header[1]
        elif header[0] == b':path':
            self.path = header[1]
        elif header[0] == b':authority':
            _addHeaderToRequest(self._request, (b'host', header[1]))
    if not gotLength:
        if self.command in (b'GET', b'HEAD'):
            self._request.gotLength(0)
        else:
            self._request.gotLength(None)
    self._request.parseCookies()
    expectContinue = self._request.requestHeaders.getRawHeaders(b'expect')
    if expectContinue and expectContinue[0].lower() == b'100-continue':
        self._send100Continue()