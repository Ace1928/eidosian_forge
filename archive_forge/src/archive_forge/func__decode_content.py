import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
def _decode_content(self, data: bytes) -> bytes:
    encoding = self.headers.get(CONTENT_ENCODING, '').lower()
    if encoding == 'identity':
        return data
    if encoding in {'deflate', 'gzip'}:
        return ZLibDecompressor(encoding=encoding, suppress_deflate_header=True).decompress_sync(data)
    raise RuntimeError(f'unknown content encoding: {encoding}')