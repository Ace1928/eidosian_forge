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
def append_json(self, obj: Any, headers: Optional[MultiMapping[str]]=None) -> Payload:
    """Helper to append JSON part."""
    if headers is None:
        headers = CIMultiDict()
    return self.append_payload(JsonPayload(obj, headers=headers))