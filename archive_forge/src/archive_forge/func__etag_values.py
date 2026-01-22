import asyncio
import datetime
import io
import re
import socket
import string
import tempfile
import types
import warnings
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import (
from urllib.parse import parse_qsl
import attr
from multidict import (
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import (
from .web_exceptions import HTTPRequestEntityTooLarge
from .web_response import StreamResponse
@staticmethod
def _etag_values(etag_header: str) -> Iterator[ETag]:
    """Extract `ETag` objects from raw header."""
    if etag_header == ETAG_ANY:
        yield ETag(is_weak=False, value=ETAG_ANY)
    else:
        for match in LIST_QUOTED_ETAG_RE.finditer(etag_header):
            is_weak, value, garbage = match.group(2, 3, 4)
            if garbage:
                break
            yield ETag(is_weak=bool(is_weak), value=value)