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
class HttpPayloadParser:

    def __init__(self, payload: StreamReader, length: Optional[int]=None, chunked: bool=False, compression: Optional[str]=None, code: Optional[int]=None, method: Optional[str]=None, readall: bool=False, response_with_body: bool=True, auto_decompress: bool=True, lax: bool=False) -> None:
        self._length = 0
        self._type = ParseState.PARSE_NONE
        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
        self._chunk_size = 0
        self._chunk_tail = b''
        self._auto_decompress = auto_decompress
        self._lax = lax
        self.done = False
        if response_with_body and compression and self._auto_decompress:
            real_payload: Union[StreamReader, DeflateBuffer] = DeflateBuffer(payload, compression)
        else:
            real_payload = payload
        if not response_with_body:
            self._type = ParseState.PARSE_NONE
            real_payload.feed_eof()
            self.done = True
        elif chunked:
            self._type = ParseState.PARSE_CHUNKED
        elif length is not None:
            self._type = ParseState.PARSE_LENGTH
            self._length = length
            if self._length == 0:
                real_payload.feed_eof()
                self.done = True
        elif readall and code != 204:
            self._type = ParseState.PARSE_UNTIL_EOF
        elif method in ('PUT', 'POST'):
            internal_logger.warning('Content-Length or Transfer-Encoding header is required')
            self._type = ParseState.PARSE_NONE
            real_payload.feed_eof()
            self.done = True
        self.payload = real_payload

    def feed_eof(self) -> None:
        if self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_eof()
        elif self._type == ParseState.PARSE_LENGTH:
            raise ContentLengthError('Not enough data for satisfy content length header.')
        elif self._type == ParseState.PARSE_CHUNKED:
            raise TransferEncodingError('Not enough data for satisfy transfer length header.')

    def feed_data(self, chunk: bytes, SEP: _SEP=b'\r\n', CHUNK_EXT: bytes=b';') -> Tuple[bool, bytes]:
        if self._type == ParseState.PARSE_LENGTH:
            required = self._length
            chunk_len = len(chunk)
            if required >= chunk_len:
                self._length = required - chunk_len
                self.payload.feed_data(chunk, chunk_len)
                if self._length == 0:
                    self.payload.feed_eof()
                    return (True, b'')
            else:
                self._length = 0
                self.payload.feed_data(chunk[:required], required)
                self.payload.feed_eof()
                return (True, chunk[required:])
        elif self._type == ParseState.PARSE_CHUNKED:
            if self._chunk_tail:
                chunk = self._chunk_tail + chunk
                self._chunk_tail = b''
            while chunk:
                if self._chunk == ChunkState.PARSE_CHUNKED_SIZE:
                    pos = chunk.find(SEP)
                    if pos >= 0:
                        i = chunk.find(CHUNK_EXT, 0, pos)
                        if i >= 0:
                            size_b = chunk[:i]
                        else:
                            size_b = chunk[:pos]
                        if self._lax:
                            size_b = size_b.strip()
                        if not re.fullmatch(HEXDIGITS, size_b):
                            exc = TransferEncodingError(chunk[:pos].decode('ascii', 'surrogateescape'))
                            self.payload.set_exception(exc)
                            raise exc
                        size = int(bytes(size_b), 16)
                        chunk = chunk[pos + len(SEP):]
                        if size == 0:
                            self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                            if self._lax and chunk.startswith(b'\r'):
                                chunk = chunk[1:]
                        else:
                            self._chunk = ChunkState.PARSE_CHUNKED_CHUNK
                            self._chunk_size = size
                            self.payload.begin_http_chunk_receiving()
                    else:
                        self._chunk_tail = chunk
                        return (False, b'')
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK:
                    required = self._chunk_size
                    chunk_len = len(chunk)
                    if required > chunk_len:
                        self._chunk_size = required - chunk_len
                        self.payload.feed_data(chunk, chunk_len)
                        return (False, b'')
                    else:
                        self._chunk_size = 0
                        self.payload.feed_data(chunk[:required], required)
                        chunk = chunk[required:]
                        if self._lax and chunk.startswith(b'\r'):
                            chunk = chunk[1:]
                        self._chunk = ChunkState.PARSE_CHUNKED_CHUNK_EOF
                        self.payload.end_http_chunk_receiving()
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK_EOF:
                    if chunk[:len(SEP)] == SEP:
                        chunk = chunk[len(SEP):]
                        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
                    else:
                        self._chunk_tail = chunk
                        return (False, b'')
                if self._chunk == ChunkState.PARSE_MAYBE_TRAILERS:
                    head = chunk[:len(SEP)]
                    if head == SEP:
                        self.payload.feed_eof()
                        return (True, chunk[len(SEP):])
                    if not head:
                        return (False, b'')
                    if head == SEP[:1]:
                        self._chunk_tail = head
                        return (False, b'')
                    self._chunk = ChunkState.PARSE_TRAILERS
                if self._chunk == ChunkState.PARSE_TRAILERS:
                    pos = chunk.find(SEP)
                    if pos >= 0:
                        chunk = chunk[pos + len(SEP):]
                        self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                    else:
                        self._chunk_tail = chunk
                        return (False, b'')
        elif self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_data(chunk, len(chunk))
        return (False, b'')