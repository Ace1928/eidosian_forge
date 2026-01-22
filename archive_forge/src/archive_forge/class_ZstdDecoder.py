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
class ZstdDecoder(ContentDecoder):

    def __init__(self) -> None:
        self._obj = zstd.ZstdDecompressor().decompressobj()

    def decompress(self, data: bytes) -> bytes:
        if not data:
            return b''
        data_parts = [self._obj.decompress(data)]
        while self._obj.eof and self._obj.unused_data:
            unused_data = self._obj.unused_data
            self._obj = zstd.ZstdDecompressor().decompressobj()
            data_parts.append(self._obj.decompress(unused_data))
        return b''.join(data_parts)

    def flush(self) -> bytes:
        ret = self._obj.flush()
        if not self._obj.eof:
            raise DecodeError('Zstandard data is incomplete')
        return ret