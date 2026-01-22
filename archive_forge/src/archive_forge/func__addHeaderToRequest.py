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
def _addHeaderToRequest(request, header):
    """
    Add a header tuple to a request header object.

    @param request: The request to add the header tuple to.
    @type request: L{twisted.web.http.Request}

    @param header: The header tuple to add to the request.
    @type header: A L{tuple} with two elements, the header name and header
        value, both as L{bytes}.

    @return: If the header being added was the C{Content-Length} header.
    @rtype: L{bool}
    """
    requestHeaders = request.requestHeaders
    name, value = header
    values = requestHeaders.getRawHeaders(name)
    if values is not None:
        values.append(value)
    else:
        requestHeaders.setRawHeaders(name, [value])
    if name == b'content-length':
        request.gotLength(int(value))
        return True
    return False