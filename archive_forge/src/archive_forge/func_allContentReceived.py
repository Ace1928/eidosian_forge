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
def allContentReceived(self):
    command = self._command
    path = self._path
    version = self._version
    self.length = 0
    self._receivedHeaderCount = 0
    self._receivedHeaderSize = 0
    self.__first_line = 1
    self._transferDecoder = None
    del self._command, self._path, self._version
    if self.timeOut:
        self._savedTimeOut = self.setTimeout(None)
    self._handlingRequest = True
    self.setRawMode()
    req = self.requests[-1]
    req.requestReceived(command, path, version)