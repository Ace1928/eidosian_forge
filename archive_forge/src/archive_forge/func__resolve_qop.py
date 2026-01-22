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
def _resolve_qop(self, qop: bytes | None, request: Request) -> bytes | None:
    if qop is None:
        return None
    qops = re.split(b', ?', qop)
    if b'auth' in qops:
        return b'auth'
    if qops == [b'auth-int']:
        raise NotImplementedError('Digest auth-int support is not yet implemented')
    message = f'Unexpected qop value "{qop!r}" in digest auth'
    raise ProtocolError(message, request=request)