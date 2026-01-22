from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
def _dataReceived_TRAILER(self) -> bool:
    """
        Collect trailer headers if received and finish at the terminal zero-length
        chunk. Then invoke C{finishCallback} and switch to state C{'FINISHED'}.

        @returns: C{False}, as there is either insufficient data to continue,
            or no data remains.
        """
    if self._receivedTrailerHeadersSize + len(self._buffer) > self._maxTrailerHeadersSize:
        raise _MalformedChunkedDataError('Trailer headers data is too long.')
    eolIndex = self._buffer.find(b'\r\n', self._start)
    if eolIndex == -1:
        return False
    if eolIndex > 0:
        self._trailerHeaders.append(self._buffer[0:eolIndex])
        del self._buffer[0:eolIndex + 2]
        self._start = 0
        self._receivedTrailerHeadersSize += eolIndex + 2
        return True
    data = memoryview(self._buffer)[2:].tobytes()
    del self._buffer[:]
    self.state = 'FINISHED'
    self.finishCallback(data)
    return False