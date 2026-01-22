import copy
import os
import re
import zlib
from binascii import hexlify
from html import escape
from typing import List, Optional
from urllib.parse import quote as _quote
from zope.interface import implementer
from incremental import Version
from twisted import copyright
from twisted.internet import address, interfaces
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.logger import Logger
from twisted.python import components, failure, reflect
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.spread.pb import Copyable, ViewPoint
from twisted.web import http, iweb, resource, util
from twisted.web.error import UnsupportedMethod
from twisted.web.http import unquote
@implementer(iweb._IRequestEncoderFactory)
class GzipEncoderFactory:
    """
    @cvar compressLevel: The compression level used by the compressor, default
        to 9 (highest).

    @since: 12.3
    """
    _gzipCheckRegex = re.compile(b'(:?^|[\\s,])gzip(:?$|[\\s,])')
    compressLevel = 9

    def encoderForRequest(self, request):
        """
        Check the headers if the client accepts gzip encoding, and encodes the
        request if so.
        """
        acceptHeaders = b','.join(request.requestHeaders.getRawHeaders(b'accept-encoding', []))
        if self._gzipCheckRegex.search(acceptHeaders):
            encoding = request.responseHeaders.getRawHeaders(b'content-encoding')
            if encoding:
                encoding = b','.join(encoding + [b'gzip'])
            else:
                encoding = b'gzip'
            request.responseHeaders.setRawHeaders(b'content-encoding', [encoding])
            return _GzipEncoder(self.compressLevel, request)