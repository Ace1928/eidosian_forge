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
def _get_boundary(self) -> str:
    mimetype = parse_mimetype(self.headers[CONTENT_TYPE])
    assert mimetype.type == 'multipart', 'multipart/* content type expected'
    if 'boundary' not in mimetype.parameters:
        raise ValueError('boundary missed for Content-Type: %s' % self.headers[CONTENT_TYPE])
    boundary = mimetype.parameters['boundary']
    if len(boundary) > 70:
        raise ValueError('boundary %r is too long (70 chars max)' % boundary)
    return boundary