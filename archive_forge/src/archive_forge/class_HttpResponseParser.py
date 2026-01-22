import abc
import asyncio
import re
import string
from contextlib import suppress
from enum import IntEnum
from typing import (
from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL
from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import HAS_BROTLI, BrotliDecompressor, ZLibDecompressor
from .helpers import (
from .http_exceptions import (
from .http_writer import HttpVersion, HttpVersion10
from .log import internal_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders
class HttpResponseParser(HttpParser[RawResponseMessage]):
    """Read response status line and headers.

    BadStatusLine could be raised in case of any errors in status line.
    Returns RawResponseMessage.
    """
    lax = not DEBUG

    def feed_data(self, data: bytes, SEP: Optional[_SEP]=None, *args: Any, **kwargs: Any) -> Tuple[List[Tuple[RawResponseMessage, StreamReader]], bool, bytes]:
        if SEP is None:
            SEP = b'\r\n' if DEBUG else b'\n'
        return super().feed_data(data, SEP, *args, **kwargs)

    def parse_message(self, lines: List[bytes]) -> RawResponseMessage:
        line = lines[0].decode('utf-8', 'surrogateescape')
        try:
            version, status = line.split(maxsplit=1)
        except ValueError:
            raise BadStatusLine(line) from None
        try:
            status, reason = status.split(maxsplit=1)
        except ValueError:
            status = status.strip()
            reason = ''
        if len(reason) > self.max_line_size:
            raise LineTooLong('Status line is too long', str(self.max_line_size), str(len(reason)))
        match = VERSRE.fullmatch(version)
        if match is None:
            raise BadStatusLine(line)
        version_o = HttpVersion(int(match.group(1)), int(match.group(2)))
        if len(status) != 3 or not DIGITS.fullmatch(status):
            raise BadStatusLine(line)
        status_i = int(status)
        headers, raw_headers, close, compression, upgrade, chunked = self.parse_headers(lines)
        if close is None:
            close = version_o <= HttpVersion10
        return RawResponseMessage(version_o, status_i, reason.strip(), headers, raw_headers, close, compression, upgrade, chunked)