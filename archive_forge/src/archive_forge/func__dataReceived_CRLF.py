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
def _dataReceived_CRLF(self) -> bool:
    """
        Await the carriage return and line feed characters that are the end of
        chunk marker that follow the chunk data.

        @returns: C{True} when the CRLF have been read, otherwise C{False}.

        @raises _MalformedChunkedDataError: when anything other than CRLF are
            received.
        """
    if len(self._buffer) < 2:
        return False
    if not self._buffer.startswith(b'\r\n'):
        raise _MalformedChunkedDataError('Chunk did not end with CRLF')
    self.state = 'CHUNK_LENGTH'
    del self._buffer[0:2]
    return True