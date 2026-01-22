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
class HttpParser(abc.ABC, Generic[_MsgT]):
    lax: ClassVar[bool] = False

    def __init__(self, protocol: Optional[BaseProtocol]=None, loop: Optional[asyncio.AbstractEventLoop]=None, limit: int=2 ** 16, max_line_size: int=8190, max_headers: int=32768, max_field_size: int=8190, timer: Optional[BaseTimerContext]=None, code: Optional[int]=None, method: Optional[str]=None, readall: bool=False, payload_exception: Optional[Type[BaseException]]=None, response_with_body: bool=True, read_until_eof: bool=False, auto_decompress: bool=True) -> None:
        self.protocol = protocol
        self.loop = loop
        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size
        self.timer = timer
        self.code = code
        self.method = method
        self.readall = readall
        self.payload_exception = payload_exception
        self.response_with_body = response_with_body
        self.read_until_eof = read_until_eof
        self._lines: List[bytes] = []
        self._tail = b''
        self._upgraded = False
        self._payload = None
        self._payload_parser: Optional[HttpPayloadParser] = None
        self._auto_decompress = auto_decompress
        self._limit = limit
        self._headers_parser = HeadersParser(max_line_size, max_headers, max_field_size)

    @abc.abstractmethod
    def parse_message(self, lines: List[bytes]) -> _MsgT:
        pass

    def feed_eof(self) -> Optional[_MsgT]:
        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            self._payload_parser = None
        else:
            if self._tail:
                self._lines.append(self._tail)
            if self._lines:
                if self._lines[-1] != '\r\n':
                    self._lines.append(b'')
                with suppress(Exception):
                    return self.parse_message(self._lines)
        return None

    def feed_data(self, data: bytes, SEP: _SEP=b'\r\n', EMPTY: bytes=b'', CONTENT_LENGTH: istr=hdrs.CONTENT_LENGTH, METH_CONNECT: str=hdrs.METH_CONNECT, SEC_WEBSOCKET_KEY1: istr=hdrs.SEC_WEBSOCKET_KEY1) -> Tuple[List[Tuple[_MsgT, StreamReader]], bool, bytes]:
        messages = []
        if self._tail:
            data, self._tail = (self._tail + data, b'')
        data_len = len(data)
        start_pos = 0
        loop = self.loop
        while start_pos < data_len:
            if self._payload_parser is None and (not self._upgraded):
                pos = data.find(SEP, start_pos)
                if pos == start_pos and (not self._lines):
                    start_pos = pos + len(SEP)
                    continue
                if pos >= start_pos:
                    line = data[start_pos:pos]
                    if SEP == b'\n':
                        line = line.rstrip(b'\r')
                    self._lines.append(line)
                    start_pos = pos + len(SEP)
                    if self._lines[-1] == EMPTY:
                        try:
                            msg: _MsgT = self.parse_message(self._lines)
                        finally:
                            self._lines.clear()

                        def get_content_length() -> Optional[int]:
                            length_hdr = msg.headers.get(CONTENT_LENGTH)
                            if length_hdr is None:
                                return None
                            if not DIGITS.fullmatch(length_hdr):
                                raise InvalidHeader(CONTENT_LENGTH)
                            return int(length_hdr)
                        length = get_content_length()
                        if SEC_WEBSOCKET_KEY1 in msg.headers:
                            raise InvalidHeader(SEC_WEBSOCKET_KEY1)
                        self._upgraded = msg.upgrade
                        method = getattr(msg, 'method', self.method)
                        code = getattr(msg, 'code', 0)
                        assert self.protocol is not None
                        empty_body = status_code_must_be_empty_body(code) or bool(method and method_must_be_empty_body(method))
                        if not empty_body and (length is not None and length > 0 or (msg.chunked and (not msg.upgrade))):
                            payload = StreamReader(self.protocol, timer=self.timer, loop=loop, limit=self._limit)
                            payload_parser = HttpPayloadParser(payload, length=length, chunked=msg.chunked, method=method, compression=msg.compression, code=self.code, readall=self.readall, response_with_body=self.response_with_body, auto_decompress=self._auto_decompress, lax=self.lax)
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        elif method == METH_CONNECT:
                            assert isinstance(msg, RawRequestMessage)
                            payload = StreamReader(self.protocol, timer=self.timer, loop=loop, limit=self._limit)
                            self._upgraded = True
                            self._payload_parser = HttpPayloadParser(payload, method=msg.method, compression=msg.compression, readall=True, auto_decompress=self._auto_decompress, lax=self.lax)
                        elif not empty_body and length is None and self.read_until_eof:
                            payload = StreamReader(self.protocol, timer=self.timer, loop=loop, limit=self._limit)
                            payload_parser = HttpPayloadParser(payload, length=length, chunked=msg.chunked, method=method, compression=msg.compression, code=self.code, readall=True, response_with_body=self.response_with_body, auto_decompress=self._auto_decompress, lax=self.lax)
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        else:
                            payload = EMPTY_PAYLOAD
                        messages.append((msg, payload))
                else:
                    self._tail = data[start_pos:]
                    data = EMPTY
                    break
            elif self._payload_parser is None and self._upgraded:
                assert not self._lines
                break
            elif data and start_pos < data_len:
                assert not self._lines
                assert self._payload_parser is not None
                try:
                    eof, data = self._payload_parser.feed_data(data[start_pos:], SEP)
                except BaseException as exc:
                    if self.payload_exception is not None:
                        self._payload_parser.payload.set_exception(self.payload_exception(str(exc)))
                    else:
                        self._payload_parser.payload.set_exception(exc)
                    eof = True
                    data = b''
                if eof:
                    start_pos = 0
                    data_len = len(data)
                    self._payload_parser = None
                    continue
            else:
                break
        if data and start_pos < data_len:
            data = data[start_pos:]
        else:
            data = EMPTY
        return (messages, self._upgraded, data)

    def parse_headers(self, lines: List[bytes]) -> Tuple['CIMultiDictProxy[str]', RawHeaders, Optional[bool], Optional[str], bool, bool]:
        """Parses RFC 5322 headers from a stream.

        Line continuations are supported. Returns list of header name
        and value pairs. Header name is in upper case.
        """
        headers, raw_headers = self._headers_parser.parse_headers(lines)
        close_conn = None
        encoding = None
        upgrade = False
        chunked = False
        singletons = (hdrs.CONTENT_LENGTH, hdrs.CONTENT_LOCATION, hdrs.CONTENT_RANGE, hdrs.CONTENT_TYPE, hdrs.ETAG, hdrs.HOST, hdrs.MAX_FORWARDS, hdrs.SERVER, hdrs.TRANSFER_ENCODING, hdrs.USER_AGENT)
        bad_hdr = next((h for h in singletons if len(headers.getall(h, ())) > 1), None)
        if bad_hdr is not None:
            raise BadHttpMessage(f"Duplicate '{bad_hdr}' header found.")
        conn = headers.get(hdrs.CONNECTION)
        if conn:
            v = conn.lower()
            if v == 'close':
                close_conn = True
            elif v == 'keep-alive':
                close_conn = False
            elif v == 'upgrade' and headers.get(hdrs.UPGRADE):
                upgrade = True
        enc = headers.get(hdrs.CONTENT_ENCODING)
        if enc:
            enc = enc.lower()
            if enc in ('gzip', 'deflate', 'br'):
                encoding = enc
        te = headers.get(hdrs.TRANSFER_ENCODING)
        if te is not None:
            if 'chunked' == te.lower():
                chunked = True
            else:
                raise BadHttpMessage('Request has invalid `Transfer-Encoding`')
            if hdrs.CONTENT_LENGTH in headers:
                raise BadHttpMessage("Transfer-Encoding can't be present with Content-Length")
        return (headers, raw_headers, close_conn, encoding, upgrade, chunked)

    def set_upgraded(self, val: bool) -> None:
        """Set connection upgraded (to websocket) mode.

        :param bool val: new state.
        """
        self._upgraded = val