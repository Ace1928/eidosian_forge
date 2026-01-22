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
def append_payload(self, payload: Payload) -> Payload:
    """Adds a new body part to multipart writer."""
    encoding: Optional[str] = payload.headers.get(CONTENT_ENCODING, '').lower()
    if encoding and encoding not in ('deflate', 'gzip', 'identity'):
        raise RuntimeError(f'unknown content encoding: {encoding}')
    if encoding == 'identity':
        encoding = None
    te_encoding: Optional[str] = payload.headers.get(CONTENT_TRANSFER_ENCODING, '').lower()
    if te_encoding not in ('', 'base64', 'quoted-printable', 'binary'):
        raise RuntimeError('unknown content transfer encoding: {}'.format(te_encoding))
    if te_encoding == 'binary':
        te_encoding = None
    size = payload.size
    if size is not None and (not (encoding or te_encoding)):
        payload.headers[CONTENT_LENGTH] = str(size)
    self._parts.append((payload, encoding, te_encoding))
    return payload