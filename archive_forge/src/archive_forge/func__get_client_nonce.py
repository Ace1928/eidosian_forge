from __future__ import annotations
import hashlib
import os
import re
import time
import typing
from base64 import b64encode
from urllib.request import parse_http_list
from ._exceptions import ProtocolError
from ._models import Cookies, Request, Response
from ._utils import to_bytes, to_str, unquote
def _get_client_nonce(self, nonce_count: int, nonce: bytes) -> bytes:
    s = str(nonce_count).encode()
    s += nonce
    s += time.ctime().encode()
    s += os.urandom(8)
    return hashlib.sha1(s).hexdigest()[:16].encode()