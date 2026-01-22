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
def _dataReceived_BODY(self) -> bool:
    """
        Deliver any available chunk data to the C{dataCallback}. When all the
        remaining data for the chunk arrives, switch to state C{'CRLF'}.

        @returns: C{True} to continue processing of any buffered data.
        """
    if len(self._buffer) >= self.length:
        chunk = memoryview(self._buffer)[:self.length].tobytes()
        del self._buffer[:self.length]
        self.state = 'CRLF'
        self.dataCallback(chunk)
    else:
        chunk = bytes(self._buffer)
        self.length -= len(chunk)
        del self._buffer[:]
        self.dataCallback(chunk)
    return True