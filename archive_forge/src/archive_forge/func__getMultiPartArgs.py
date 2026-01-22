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
def _getMultiPartArgs(content: bytes, ctype: bytes) -> dict[bytes, list[bytes]]:
    """
    Parse the content of a multipart/form-data request.
    """
    result = {}
    multiPartHeaders = b'MIME-Version: 1.0\r\n' + b'Content-Type: ' + ctype + b'\r\n'
    msg = message_from_bytes(multiPartHeaders + content)
    if not msg.is_multipart():
        raise _MultiPartParseException('Not a multipart.')
    for part in msg.get_payload():
        name = part.get_param('name', header='content-disposition')
        if not name:
            continue
        payload = part.get_payload(decode=True)
        result[name.encode('utf8')] = [payload]
    return result