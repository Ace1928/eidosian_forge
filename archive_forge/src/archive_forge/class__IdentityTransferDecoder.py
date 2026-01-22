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
class _IdentityTransferDecoder:
    """
    Protocol for accumulating bytes up to a specified length.  This handles the
    case where no I{Transfer-Encoding} is specified.

    @ivar contentLength: Counter keeping track of how many more bytes there are
        to receive.

    @ivar dataCallback: A one-argument callable which will be invoked each
        time application data is received.

    @ivar finishCallback: A one-argument callable which will be invoked when
        the terminal chunk is received.  It will be invoked with all bytes
        which were delivered to this protocol which came after the terminal
        chunk.
    """

    def __init__(self, contentLength, dataCallback, finishCallback):
        self.contentLength = contentLength
        self.dataCallback = dataCallback
        self.finishCallback = finishCallback

    def dataReceived(self, data):
        """
        Interpret the next chunk of bytes received.  Either deliver them to the
        data callback or invoke the finish callback if enough bytes have been
        received.

        @raise RuntimeError: If the finish callback has already been invoked
            during a previous call to this methood.
        """
        if self.dataCallback is None:
            raise RuntimeError('_IdentityTransferDecoder cannot decode data after finishing')
        if self.contentLength is None:
            self.dataCallback(data)
        elif len(data) < self.contentLength:
            self.contentLength -= len(data)
            self.dataCallback(data)
        else:
            contentLength = self.contentLength
            dataCallback = self.dataCallback
            finishCallback = self.finishCallback
            self.dataCallback = self.finishCallback = None
            self.contentLength = 0
            dataCallback(data[:contentLength])
            finishCallback(data[contentLength:])

    def noMoreData(self):
        """
        All data which will be delivered to this decoder has been.  Check to
        make sure as much data as was expected has been received.

        @raise PotentialDataLoss: If the content length is unknown.
        @raise _DataLoss: If the content length is known and fewer than that
            many bytes have been delivered.

        @return: L{None}
        """
        finishCallback = self.finishCallback
        self.dataCallback = self.finishCallback = None
        if self.contentLength is None:
            finishCallback(b'')
            raise PotentialDataLoss()
        elif self.contentLength != 0:
            raise _DataLoss()