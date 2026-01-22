from __future__ import annotations
import collections
import io
import json as _json
import logging
import re
import sys
import typing
import warnings
import zlib
from contextlib import contextmanager
from http.client import HTTPMessage as _HttplibHTTPMessage
from http.client import HTTPResponse as _HttplibHTTPResponse
from socket import timeout as SocketTimeout
from . import util
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPConnection, HTTPException
from .exceptions import (
from .util.response import is_fp_closed, is_response_to_head
from .util.retry import Retry
class BaseHTTPResponse(io.IOBase):
    CONTENT_DECODERS = ['gzip', 'x-gzip', 'deflate']
    if brotli is not None:
        CONTENT_DECODERS += ['br']
    if zstd is not None:
        CONTENT_DECODERS += ['zstd']
    REDIRECT_STATUSES = [301, 302, 303, 307, 308]
    DECODER_ERROR_CLASSES: tuple[type[Exception], ...] = (IOError, zlib.error)
    if brotli is not None:
        DECODER_ERROR_CLASSES += (brotli.error,)
    if zstd is not None:
        DECODER_ERROR_CLASSES += (zstd.ZstdError,)

    def __init__(self, *, headers: typing.Mapping[str, str] | typing.Mapping[bytes, bytes] | None=None, status: int, version: int, reason: str | None, decode_content: bool, request_url: str | None, retries: Retry | None=None) -> None:
        if isinstance(headers, HTTPHeaderDict):
            self.headers = headers
        else:
            self.headers = HTTPHeaderDict(headers)
        self.status = status
        self.version = version
        self.reason = reason
        self.decode_content = decode_content
        self._has_decoded_content = False
        self._request_url: str | None = request_url
        self.retries = retries
        self.chunked = False
        tr_enc = self.headers.get('transfer-encoding', '').lower()
        encodings = (enc.strip() for enc in tr_enc.split(','))
        if 'chunked' in encodings:
            self.chunked = True
        self._decoder: ContentDecoder | None = None
        self.length_remaining: int | None

    def get_redirect_location(self) -> str | None | Literal[False]:
        """
        Should we redirect and where to?

        :returns: Truthy redirect location string if we got a redirect status
            code and valid location. ``None`` if redirect status and no
            location. ``False`` if not a redirect status code.
        """
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get('location')
        return False

    @property
    def data(self) -> bytes:
        raise NotImplementedError()

    def json(self) -> typing.Any:
        """
        Parses the body of the HTTP response as JSON.

        To use a custom JSON decoder pass the result of :attr:`HTTPResponse.data` to the decoder.

        This method can raise either `UnicodeDecodeError` or `json.JSONDecodeError`.

        Read more :ref:`here <json>`.
        """
        data = self.data.decode('utf-8')
        return _json.loads(data)

    @property
    def url(self) -> str | None:
        raise NotImplementedError()

    @url.setter
    def url(self, url: str | None) -> None:
        raise NotImplementedError()

    @property
    def connection(self) -> BaseHTTPConnection | None:
        raise NotImplementedError()

    @property
    def retries(self) -> Retry | None:
        return self._retries

    @retries.setter
    def retries(self, retries: Retry | None) -> None:
        if retries is not None and retries.history:
            self.url = retries.history[-1].redirect_location
        self._retries = retries

    def stream(self, amt: int | None=2 ** 16, decode_content: bool | None=None) -> typing.Iterator[bytes]:
        raise NotImplementedError()

    def read(self, amt: int | None=None, decode_content: bool | None=None, cache_content: bool=False) -> bytes:
        raise NotImplementedError()

    def read1(self, amt: int | None=None, decode_content: bool | None=None) -> bytes:
        raise NotImplementedError()

    def read_chunked(self, amt: int | None=None, decode_content: bool | None=None) -> typing.Iterator[bytes]:
        raise NotImplementedError()

    def release_conn(self) -> None:
        raise NotImplementedError()

    def drain_conn(self) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def _init_decoder(self) -> None:
        """
        Set-up the _decoder attribute if necessary.
        """
        content_encoding = self.headers.get('content-encoding', '').lower()
        if self._decoder is None:
            if content_encoding in self.CONTENT_DECODERS:
                self._decoder = _get_decoder(content_encoding)
            elif ',' in content_encoding:
                encodings = [e.strip() for e in content_encoding.split(',') if e.strip() in self.CONTENT_DECODERS]
                if encodings:
                    self._decoder = _get_decoder(content_encoding)

    def _decode(self, data: bytes, decode_content: bool | None, flush_decoder: bool) -> bytes:
        """
        Decode the data passed in and potentially flush the decoder.
        """
        if not decode_content:
            if self._has_decoded_content:
                raise RuntimeError('Calling read(decode_content=False) is not supported after read(decode_content=True) was called.')
            return data
        try:
            if self._decoder:
                data = self._decoder.decompress(data)
                self._has_decoded_content = True
        except self.DECODER_ERROR_CLASSES as e:
            content_encoding = self.headers.get('content-encoding', '').lower()
            raise DecodeError('Received response with content-encoding: %s, but failed to decode it.' % content_encoding, e) from e
        if flush_decoder:
            data += self._flush_decoder()
        return data

    def _flush_decoder(self) -> bytes:
        """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
        if self._decoder:
            return self._decoder.decompress(b'') + self._decoder.flush()
        return b''

    def readinto(self, b: bytearray) -> int:
        temp = self.read(len(b))
        if len(temp) == 0:
            return 0
        else:
            b[:len(temp)] = temp
            return len(temp)

    def getheaders(self) -> HTTPHeaderDict:
        warnings.warn('HTTPResponse.getheaders() is deprecated and will be removed in urllib3 v2.1.0. Instead access HTTPResponse.headers directly.', category=DeprecationWarning, stacklevel=2)
        return self.headers

    def getheader(self, name: str, default: str | None=None) -> str | None:
        warnings.warn('HTTPResponse.getheader() is deprecated and will be removed in urllib3 v2.1.0. Instead use HTTPResponse.headers.get(name, default).', category=DeprecationWarning, stacklevel=2)
        return self.headers.get(name, default)

    def info(self) -> HTTPHeaderDict:
        return self.headers

    def geturl(self) -> str | None:
        return self.url